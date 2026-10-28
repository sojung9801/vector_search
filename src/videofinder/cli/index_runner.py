"""vf-index <영상.mp4> — 영상 한 개를 색인한다."""

from __future__ import annotations

import argparse

from videofinder.pipeline.indexer import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="영상 한 개를 발화·화면 검색용으로 색인")
    parser.add_argument("video")
    parser.add_argument("--interval", type=float, default=1.0, help="프레임 샘플링 간격(초)")
    args = parser.parse_args()
    print(build_index(args.video, args.interval))


if __name__ == "__main__":
    main()
