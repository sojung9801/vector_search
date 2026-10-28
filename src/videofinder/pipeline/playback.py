"""브라우저 재생용 사본. 원본이 모든 브라우저에서 재생되는 조합(mp4 + H.264 + AAC/MP3)이 아니면 만든다.

예: webm(vp8 + Vorbis)은 Safari에서 재생되지 않을 수 있다. 분석에는 원본을 쓰고 재생에만 사본을 쓴다.
"""

from __future__ import annotations

from pathlib import Path

from videofinder.pipeline.media import MediaInfo, run_ffmpeg

PLAYBACK_FILE = "playback.mp4"
BROWSER_SAFE_AUDIO = {"aac", "mp3"}


def needs_playback_copy(info: MediaInfo) -> bool:
    is_mp4 = "mp4" in info.format_name.split(",")
    audio_ok = info.audio_codec is None or info.audio_codec in BROWSER_SAFE_AUDIO
    return not (is_mp4 and info.video_codec == "h264" and audio_ok)


def make_playback_copy(source: str | Path, out: Path) -> Path:
    """720p(원본이 더 작으면 원본 크기) H.264/AAC mp4. 12분 1080p webm 기준 약 1분, 약 120MB."""
    tmp = out.with_name(out.name + ".part.mp4")
    run_ffmpeg(
        [
            "-i", str(source),
            "-vf", "scale=-2:'min(720,ih)'",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(tmp),
        ],
        "재생용 사본 만들기",
    )
    tmp.replace(out)
    return out
