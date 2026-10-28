"""vf-bench-search — 검색 서버 경량화 측정: 로딩 시간·검색 시간·결과(시각·점수)를 기록한다 (내용 미출력).

  vf-bench-search out.json [--half] [--text-only-siglip]
두 결과 파일을 --compare a.json b.json 으로 비교하면 순위가 얼마나 바뀌었는지 나온다.
메모리는 /usr/bin/time -l 로 감싸 peak memory footprint 를 본다.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

import videofinder.pipeline.encoders as encoders
from videofinder.pipeline.jsonio import read_json, write_json

QUERIES = [
    ("a125b0de967b562a", "꼭짓점 구하는 법"), ("a125b0de967b562a", "아래로 볼록"), ("a125b0de967b562a", "그래프 그리는 앱"),
    ("a125b0de967b562a", "칠판에 그래프를 그리는 장면"), ("a125b0de967b562a", "선생님이 두 손을 든 모습"), ("a125b0de967b562a", "오늘 저녁 메뉴"),
    ("328253280a05e6b1", "총 쏘는 장면"), ("328253280a05e6b1", "로봇이 나오는 장면"), ("328253280a05e6b1", "두 사람이 대화하는 장면"),
    ("328253280a05e6b1", "다리 위에 서 있는 사람"), ("328253280a05e6b1", "a giant robot"), ("328253280a05e6b1", "헤어지자는 말"),
    ("328253280a05e6b1", "사과하는 말"), ("328253280a05e6b1", "강아지가 뛰어노는 장면"), ("9e3a1adccdc35594", "토끼"),
    ("9e3a1adccdc35594", "나비가 날아가는 장면"), ("9e3a1adccdc35594", "사과가 떨어지는 장면"), ("9e3a1adccdc35594", "새가 나는 장면"),
    ("a125b0de967b562a", "이차항 계수의 절댓값이 클수록"), ("328253280a05e6b1", "컴퓨터 화면을 보는 사람"),
]


def bench(out: Path, half: bool, text_only_siglip: bool) -> None:
    from videofinder.app.services.video_finder import VideoFinder
    from videofinder.retrieval.search import Searcher

    started = time.perf_counter()
    finder = VideoFinder.__new__(VideoFinder)
    finder.library_dir = __import__("videofinder.pipeline.paths", fromlist=["videos_dir"]).videos_dir()
    finder._indexes = {}
    searcher = Searcher.__new__(Searcher)
    searcher.text = encoders.TextEncoder(half=half)
    searcher.reranker = encoders.Reranker(half=half)
    searcher.siglip = encoders.SiglipEncoder(text_only=text_only_siglip, half=half)
    finder._searcher = searcher
    load_s = time.perf_counter() - started
    finder.search(QUERIES[0][0], QUERIES[0][1], 8)  # 첫 검색 예열
    results: list[dict[str, Any]] = []
    times: list[float] = []
    for video_id, query in QUERIES:
        t = time.perf_counter()
        r = finder.search(video_id, query, 8)
        times.append(time.perf_counter() - t)
        results.append({"video": video_id, "query": query, **{k: [[h.start_ms, round(h.score, 5), h.low] for h in hits] for k, hits in r.items()}})
    import psutil
    import torch

    rss_gb = psutil.Process().memory_info().rss / 1e9
    gpu_gb = torch.mps.driver_allocated_memory() / 1e9 if torch.backends.mps.is_available() else 0.0
    print(f"상주 메모리: 프로세스 {rss_gb:.2f}GB + GPU {gpu_gb:.2f}GB")
    write_json(out, {"rss_gb": round(rss_gb, 2), "gpu_gb": round(gpu_gb, 2), "load_s": round(load_s, 1), "search_ms_mean": round(1000 * float(np.mean(times))), "search_ms_max": round(1000 * float(np.max(times))), "results": results})
    print(f"로딩 {load_s:.1f}초 · 검색 평균 {1000 * np.mean(times):.0f}ms · 최대 {1000 * np.max(times):.0f}ms")


def compare(a_path: Path, b_path: Path) -> None:
    a, b = read_json(a_path), read_json(b_path)
    print(f"로딩 {a['load_s']}→{b['load_s']}초 · 검색 평균 {a['search_ms_mean']}→{b['search_ms_mean']}ms · 최대 {a['search_ms_max']}→{b['search_ms_max']}ms")
    for kind in ("speech", "shots", "scenes"):
        same_top1 = same_set = total = 0
        diffs: list[float] = []
        tier_flips = 0
        for ra, rb in zip(a["results"], b["results"]):
            ha, hb = ra[kind], rb[kind]
            if not ha and not hb:
                continue
            total += 1
            same_top1 += bool(ha and hb and ha[0][0] == hb[0][0])
            same_set += {h[0] for h in ha} == {h[0] for h in hb}
            sb = {h[0]: (h[1], h[2]) for h in hb}
            for t, score, low in ha:
                if t in sb:
                    diffs.append(abs(score - sb[t][0]))
                    tier_flips += low != sb[t][1]
        if total:
            print(f"{kind:6s} 질의 {total}개 | 1위 같음 {same_top1}/{total} | 결과 집합 같음 {same_set}/{total} | 점수 차 최대 {max(diffs, default=0):.5f} | 접힘 여부 바뀜 {tier_flips}")


def main() -> None:
    parser = argparse.ArgumentParser(description="검색 경량화 측정 (내용 미출력)")
    parser.add_argument("out", nargs="?")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--text-only-siglip", action="store_true")
    parser.add_argument("--compare", nargs=2)
    parser.add_argument("--candidates", type=int, help="재정렬 후보 수 (기본: retrieval.search.RERANK_CANDIDATES)")
    parser.add_argument("--rerank-batch", type=int, help="재정렬 한 번에 처리할 개수")
    args = parser.parse_args()
    if args.compare:
        compare(Path(args.compare[0]), Path(args.compare[1]))
        return
    if args.rerank_batch:
        encoders.RERANK_BATCH = args.rerank_batch
    if args.candidates:
        import videofinder.retrieval.search as search_module

        search_module.RERANK_CANDIDATES = args.candidates
    bench(Path(args.out), args.half, args.text_only_siglip)


if __name__ == "__main__":
    main()
