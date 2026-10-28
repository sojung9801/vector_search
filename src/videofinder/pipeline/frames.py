"""화면 프레임 샘플링. FFmpeg로 일정 간격 프레임을 정사각형(레터박스) RGB로 직접 읽는다 (임시 파일 없음)."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from videofinder.pipeline.media import MediaError, resolve_ffmpeg

FRAME_SIZE = 256  # CLIP 입력(224) 보다 약간 크게. 가로가 긴 영상도 중앙 크롭으로 잘리지 않게 패딩한다.


@dataclass(frozen=True)
class Frame:
    t_ms: int
    rgb: np.ndarray  # (FRAME_SIZE, FRAME_SIZE, 3) uint8


def iter_frames(video: str | Path, interval_s: float = 1.0, size: int = FRAME_SIZE, start_s: float = 0.0, duration_s: float | None = None) -> Iterator[Frame]:
    """start_s 부터 (duration_s 동안) interval_s 간격 프레임. Frame.t_ms 는 영상 처음 기준 시각."""
    tools = resolve_ffmpeg()
    vf = f"fps=1/{interval_s:g},scale={size}:{size}:force_original_aspect_ratio=decrease,pad={size}:{size}:(ow-iw)/2:(oh-ih)/2"
    seek = ["-ss", f"{start_s:.3f}"] if start_s > 0 else []
    limit = ["-t", f"{duration_s:.3f}"] if duration_s is not None else []
    proc = subprocess.Popen(
        [tools.ffmpeg, "-nostdin", "-loglevel", "error", *seek, "-i", str(video), *limit, "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None and proc.stderr is not None
    frame_bytes = size * size * 3
    i = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield Frame(int(round((start_s + i * interval_s) * 1000)), np.frombuffer(buf, dtype=np.uint8).reshape(size, size, 3))
            i += 1
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode(errors="replace")
        proc.wait()
    if proc.returncode != 0:
        raise MediaError(f"프레임 추출 실패: {err[-300:]}")
