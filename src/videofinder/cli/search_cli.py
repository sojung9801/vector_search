"""vf-search <색인 디렉터리> <질의...> — 터미널에서 검색 결과 확인."""

from __future__ import annotations

import argparse
from pathlib import Path

from videofinder.pipeline.timecode import ms_to_timecode
from videofinder.retrieval.search import Searcher, VideoIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="색인된 영상 검색")
    parser.add_argument("index_dir")
    parser.add_argument("queries", nargs="+")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()
    searcher = Searcher()
    index = VideoIndex(Path(args.index_dir))
    for query in args.queries:
        result = searcher.search(index, query, args.k)
        print(f"\n## {query}")
        for hit in result["speech"]:
            print(f"  발화 {ms_to_timecode(hit.start_ms)} {hit.score:.3f} {hit.text}")
        for hit in result["shots"]:
            print(f"  화면 {ms_to_timecode(hit.start_ms)}~{ms_to_timecode(hit.end_ms)} {hit.score:.3f}")


if __name__ == "__main__":
    main()
