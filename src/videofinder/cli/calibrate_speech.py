"""vf-calibrate-speech — 발화 재정렬 기준값(SPEECH_MIN_PROB) 점검. 사람이 만든 정답 없이 돌릴 수 있다.

- 있는 말(positive): 영상 자기 자막에서 뽑은 문장 → 그 문장이 들어 있는 시점이 결과에 나와야 함
- 없는 말(negative): 다른 영상 자막에서 뽑은 문장 → 결과가 0개여야 함

자막 원문은 프로그램 안에서만 쓰고 출력하지 않는다. 출력은 통과율과 점수 분포뿐이다.
문장을 그대로 검색하므로 실제 사용자 검색보다 쉬운 시험이다 (표현이 다른 질의는 검증하지 못함).
"""

from __future__ import annotations

import argparse
import random

import numpy as np

import videofinder.retrieval.search as search_module
from videofinder.app.services.video_finder import VideoFinder
from videofinder.retrieval.search import Searcher, VideoIndex, is_short

MIN_CHARS = 8  # 이보다 짧은 문장은 질의로 쓰지 않음 ("네.", "아시겠습니까?" 류)
TOLERANCE_MS = 3_000
THRESHOLDS = [0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1]


def _sample(index: VideoIndex, n: int, rng: random.Random) -> list[dict[str, object]]:
    usable = [s for s in index.segments if len(str(s["text"]).strip()) >= MIN_CHARS and not is_short(str(s["text"]))]
    return rng.sample(usable, min(n, len(usable)))


def _top_prob_and_found(searcher: Searcher, index: VideoIndex, query: str, target_ms: int | None) -> tuple[float, bool]:
    """기준값 없이(0) 검색해 1위 확률과, target 시점 포함 구간이 결과에 있는지."""
    hits = searcher.search(index, query, k=8)["speech"]
    top = hits[0].score if hits else 0.0
    if target_ms is None:
        return top, False
    found_prob = next((h.score for h in hits if h.start_ms - TOLERANCE_MS <= target_ms <= h.end_ms + TOLERANCE_MS), None)
    return (found_prob if found_prob is not None else 0.0), found_prob is not None


def main() -> None:
    parser = argparse.ArgumentParser(description="발화 기준값 점검 (내용 미출력)")
    parser.add_argument("-n", type=int, default=30, help="영상당 질의 수")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    search_module.SPEECH_MIN_PROB = 0.0  # 기준값 없이 원점수를 모은 뒤 여러 기준값을 비교한다
    finder = VideoFinder()
    indexes = [finder.get(v["id"]) for v in finder.list_videos() if v["speech_segments"] >= 20]
    searcher = Searcher()
    print(f"영상 {len(indexes)}개: " + ", ".join(f"{i.id}(발화 {len(i.segments)})" for i in indexes))

    pos_probs: list[float] = []
    pos_found: list[bool] = []
    neg_probs: list[float] = []
    for index in indexes:
        for seg in _sample(index, args.n, rng):
            prob, found = _top_prob_and_found(searcher, index, str(seg["text"]), int(seg["start_ms"]))
            pos_probs.append(prob)
            pos_found.append(found)
        others = [o for o in indexes if o.id != index.id]
        for other in others:
            for seg in _sample(other, args.n // max(1, len(others)), rng):
                prob, _ = _top_prob_and_found(searcher, index, str(seg["text"]), None)
                neg_probs.append(prob)

    pos = np.array(pos_probs)
    neg = np.array(neg_probs)
    found = np.array(pos_found)
    print(f"\n있는 말 {len(pos)}개 — 기준값 없이 정답 구간이 상위 8개 안: {found.mean():.0%}")
    print(f"없는 말 {len(neg)}개")
    q = [10, 25, 50, 75, 90]
    print("있는 말 정답 확률 백분위 " + " ".join(f"p{p}={v:.3f}" for p, v in zip(q, np.percentile(pos, q))))
    print("없는 말 1위 확률 백분위 " + " ".join(f"p{p}={v:.4f}" for p, v in zip(q, np.percentile(neg, q))) + f" max={neg.max():.4f}")
    print("\n기준값 | 있는 말 통과(찾음) | 없는 말 거부(0개)")
    for t in THRESHOLDS:
        print(f"{t:6.3f} | {((pos >= t) & found).mean():6.0%} | {(neg < t).mean():6.0%}")


if __name__ == "__main__":
    main()
