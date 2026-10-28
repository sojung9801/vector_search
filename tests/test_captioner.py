import numpy as np

from videofinder.pipeline.captioner import group_clips
from videofinder.pipeline.frames import Frame


def frames(n: int, fps: int) -> list[Frame]:
    return [Frame(int(i * 1000 / fps), np.zeros((2, 2, 3), dtype=np.uint8)) for i in range(n)]


def clip_shapes(n_frames: int) -> list[tuple[int, int]]:
    return [(start, len(f)) for start, f in group_clips(frames(n_frames, 4), clip_s=4, fps=4)]


def test_short_tail_is_merged_into_previous_clip() -> None:
    # 29.5초 = 118프레임 → 4초(16) ×7 + 꼬리 6프레임(1.5초) → 마지막 구간에 합쳐 22프레임
    assert clip_shapes(118)[-2:] == [(20000, 16), (24000, 22)]


def test_long_tail_stays_separate() -> None:
    # 꼬리 10프레임(2.5초)은 따로 둔다
    assert clip_shapes(42) == [(0, 16), (4000, 16), (8000, 10)]
