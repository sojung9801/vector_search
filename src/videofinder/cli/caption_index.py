"""vf-caption <색인 id> — 이미 색인된 영상에 구간 장면 설명(captions.json/.npy)을 추가한다 (실험).

설명 문장은 영상 내용이므로 출력하지 않는다. 출력은 구간 수와 처리 시간뿐이다.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from videofinder.pipeline import paths
from videofinder.pipeline.captioner import CLIP_FPS, CLIP_S, ClipCaptioner, iter_clips
from videofinder.pipeline.jsonio import write_json
from videofinder.retrieval.search import VideoIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="구간 장면 설명 추가 (내용 미출력)")
    parser.add_argument("video_id")
    args = parser.parse_args()

    from videofinder.pipeline.encoders import TextEncoder

    video = VideoIndex(paths.videos_dir() / args.video_id)
    duration_ms = int(video.meta["video"]["duration_ms"])
    total = -(-duration_ms // (CLIP_S * 1000))

    started = time.perf_counter()
    captioner = ClipCaptioner()
    load_s = time.perf_counter() - started
    records = []
    for n, (start_ms, frames) in enumerate(iter_clips(video.video_path), start=1):
        text = captioner.caption(frames)
        end_ms = min(start_ms + len(frames) * 1000 // CLIP_FPS, duration_ms)
        records.append({"start_ms": start_ms, "end_ms": end_ms, "text": text})
        if n % 10 == 0 or n == total:
            print(f"  구간 {n}/{total} ({time.perf_counter() - started - load_s:.0f}초)")
    caption_s = time.perf_counter() - started - load_s

    write_json(video.dir / "captions.json", records)
    np.save(video.dir / "captions.npy", TextEncoder().passages([r["text"] for r in records]) if records else np.zeros((0, 768), dtype=np.float32))
    duration_s = video.meta["video"]["duration_ms"] / 1000
    print(f"완료: 구간 {len(records)}개, 모델 로딩 {load_s:.0f}초, 설명 생성 {caption_s:.0f}초 (영상 1분당 {caption_s / duration_s * 60:.0f}초)")


if __name__ == "__main__":
    main()
