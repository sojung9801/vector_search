"""FFmpeg/ffprobe 실행 계층. 앱 번들 또는 프로젝트 vendor/ 의 고정 바이너리만 사용한다 (시스템 PATH 미사용)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_VIDEO_CODECS = {"h264", "hevc", "mpeg4", "av1", "vp8", "vp9"}
SUPPORTED_AUDIO_CODECS = {"aac", "mp3", "alac", "ac3", "eac3", "opus", "vorbis", "pcm_s16le"}
# 확장자 → ffprobe format_name 에 포함돼야 하는 이름
SUPPORTED_CONTAINERS = {".mp4": "mp4", ".webm": "webm"}


class MediaError(RuntimeError):
    """FFmpeg 실행 실패 또는 미지원 미디어 (사용자에게 그대로 보여줄 한국어 메시지)."""


@dataclass(frozen=True)
class FFmpegTools:
    ffmpeg: str
    ffprobe: str
    version: str
    configuration: str


_TOOLS: FFmpegTools | None = None


def _candidate_dirs() -> list[Path]:
    dirs: list[Path] = []
    env = os.environ.get("VF_FFMPEG_DIR")
    if env:
        dirs.append(Path(env))
    exe = Path(sys.executable).resolve()
    # 앱 번들: Contents/Resources/python/bin/python3 → Contents/Resources/bin
    for parent in exe.parents:
        if parent.name == "Resources":
            dirs.append(parent / "bin")
            break
    dirs.append(Path(__file__).resolve().parents[3] / "vendor" / "ffmpeg")
    return dirs


def resolve_ffmpeg() -> FFmpegTools:
    global _TOOLS
    if _TOOLS is not None:
        return _TOOLS
    for directory in _candidate_dirs():
        ffmpeg, ffprobe = directory / "ffmpeg", directory / "ffprobe"
        if ffmpeg.is_file() and ffprobe.is_file() and os.access(ffmpeg, os.X_OK):
            proc = subprocess.run([str(ffmpeg), "-version"], capture_output=True, text=True, check=False)
            text = proc.stdout or ""
            version = re.search(r"ffmpeg version (\S+)", text)
            config = re.search(r"configuration: (.*)", text)
            _TOOLS = FFmpegTools(str(ffmpeg), str(ffprobe), version.group(1) if version else "?", config.group(1) if config else "")
            return _TOOLS
    raise MediaError("내장 FFmpeg를 찾을 수 없습니다. 앱을 다시 설치하거나 vendor/ffmpeg 를 확인하세요.")


@dataclass(frozen=True)
class MediaInfo:
    path: str
    size_bytes: int
    duration_ms: int
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: str | None
    audio_channels: int
    audio_sample_rate: int
    format_name: str

    @property
    def has_audio(self) -> bool:
        return self.audio_codec is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "duration_ms": self.duration_ms,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 3),
            "video_codec": self.video_codec,
            "audio_codec": self.audio_codec,
            "audio_channels": self.audio_channels,
            "audio_sample_rate": self.audio_sample_rate,
            "format_name": self.format_name,
        }


def _fps(text: str) -> float:
    num, _, den = text.partition("/")
    try:
        return float(num) / float(den) if den and float(den) else float(num)
    except ValueError:
        return 0.0


def probe(path: str | Path) -> MediaInfo:
    video = Path(path).expanduser()
    if not video.is_file():
        raise MediaError(f"영상 파일이 없습니다: {video}")
    tools = resolve_ffmpeg()
    proc = subprocess.run(
        [tools.ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(video)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise MediaError(f"영상 정보를 읽을 수 없습니다 (손상되었거나 지원하지 않는 형식): {proc.stderr.strip()[:300]}")
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams", [])
    vs = next((s for s in streams if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")), None)
    if vs is None:
        raise MediaError("영상 스트림이 없는 파일입니다.")
    aus = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = data.get("format", {})
    duration = float(fmt.get("duration") or vs.get("duration") or 0)
    if duration <= 0:
        raise MediaError("영상 길이를 확인할 수 없습니다.")
    return MediaInfo(
        path=str(video.resolve()),
        size_bytes=int(fmt.get("size") or video.stat().st_size),
        duration_ms=int(round(duration * 1000)),
        width=int(vs.get("width") or 0),
        height=int(vs.get("height") or 0),
        fps=_fps(str(vs.get("avg_frame_rate") or vs.get("r_frame_rate") or "0")),
        video_codec=str(vs.get("codec_name") or "unknown"),
        audio_codec=str(aus["codec_name"]) if aus else None,
        audio_channels=int(aus.get("channels") or 0) if aus else 0,
        audio_sample_rate=int(aus.get("sample_rate") or 0) if aus else 0,
        format_name=str(fmt.get("format_name") or ""),
    )


def validate_video(path: str | Path) -> MediaInfo:
    """입력 검사: 확장자(MP4·WebM), 컨테이너, 비디오/오디오 코덱."""
    video = Path(path)
    container = SUPPORTED_CONTAINERS.get(video.suffix.lower())
    if container is None:
        raise MediaError(f"MP4·WebM 파일만 지원합니다 (선택한 파일: {video.suffix or '확장자 없음'}).")
    info = probe(video)
    if container not in info.format_name.split(","):
        raise MediaError(f"{container.upper()} 컨테이너가 아닙니다 ({info.format_name}).")
    if info.video_codec not in SUPPORTED_VIDEO_CODECS:
        raise MediaError(f"지원하지 않는 영상 코덱입니다: {info.video_codec} (지원: {', '.join(sorted(SUPPORTED_VIDEO_CODECS))})")
    if info.audio_codec is not None and info.audio_codec not in SUPPORTED_AUDIO_CODECS:
        raise MediaError(f"지원하지 않는 오디오 코덱입니다: {info.audio_codec}")
    return info


def sha256_file(path: str | Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def free_disk_bytes(path: str | Path) -> int:
    target = Path(path)
    while not target.exists():
        target = target.parent
    return shutil.disk_usage(target).free


def run_ffmpeg(args: list[str], what: str) -> subprocess.CompletedProcess[str]:
    tools = resolve_ffmpeg()
    proc = subprocess.run([tools.ffmpeg, "-hide_banner", "-nostdin", "-y", *args], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-6:])
        raise MediaError(f"FFmpeg {what} 실패: {tail}")
    return proc


def read_audio_pcm(video: str | Path, start_s: float, duration_s: float, sample_rate: int = 16000) -> "np.ndarray":  # noqa: F821
    """구간 오디오를 16kHz mono float32로 직접 읽는다 (임시 WAV 파일 없음)."""
    import numpy as np

    tools = resolve_ffmpeg()
    proc = subprocess.run(
        [
            tools.ffmpeg, "-nostdin", "-loglevel", "error",
            "-ss", f"{max(0.0, start_s):.3f}", "-t", f"{duration_s:.3f}", "-i", str(video),
            "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "-",
        ],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise MediaError(f"오디오 추출 실패: {proc.stderr.decode(errors='replace')[-300:]}")
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
