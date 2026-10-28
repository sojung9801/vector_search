"""vf-eval [queries.json] — 정답 구간 기준 검색 평가.

출력은 질의·순위·점수·통과 여부뿐이다. 영상 내용(자막·썸네일)은 출력하지 않는다.

지표
- 정답 질의: Hit@1, Hit@5 (정답 구간 ±TOLERANCE_MS 안에서 시작하는 결과가 상위 N개 안에 있는가)
- 없는 질의(answers=[]): 거부율 (결과 0개인 비율)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from videofinder.app.services.video_finder import resolve_index
from videofinder.pipeline import paths
from videofinder.pipeline.jsonio import read_json, read_jsonl
from videofinder.pipeline.timecode import timecode_to_ms
from videofinder.retrieval.search import Hit, Searcher, VideoIndex

TOLERANCE_MS = 3_000
DEFAULT_QUERIES = paths.PROJECT_ROOT / "eval" / "queries.json"
DEFAULT_FEEDBACK = paths.PROJECT_ROOT / "eval" / "feedback.jsonl"


@dataclass(frozen=True)
class Outcome:
    video: str
    query: str
    target: str
    negative: bool
    rank: int | None  # 정답 질의: 첫 정답 순위(1부터), 못 찾으면 None
    returned: int
    top_score: float | None

    @property
    def passed(self) -> bool:
        return self.returned == 0 if self.negative else self.rank == 1


def _hits_for(result: dict[str, list[Hit]], target: str) -> list[Hit]:
    if target == "any":
        return sorted(result["speech"] + result["shots"] + result.get("scenes", []), key=lambda h: -h.score)
    return result[target]


def spec_from_feedback(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """검색 화면의 정답 표시(eval/feedback.jsonl) → 평가 질의.

    (영상, 검색어)마다 answer 시각과 correct 결과 구간을 정답으로 모은다.
    정답이 하나도 없고 none 표시가 있으면 '없는 질의'. wrong 표시는 정답에 넣지 않는다.
    같은 결과를 correct 로 했다가 wrong 으로 바꾸면 마지막 표시만 유효하다.
    """
    last: dict[tuple[str, str, str, int | None], dict[str, Any]] = {}
    for r in rows:
        if not str(r["query"]).strip():
            continue  # 검색 전에 눌린 기록
        key = (r["video_id"], r["query"], r.get("kind") or r["label"], r.get("start_ms"))
        last[key] = r
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for r in last.values():
        g = grouped.setdefault((r["video_id"], r["query"]), {"answers": [], "none": False})
        if r["label"] in ("answer", "correct"):
            g["answers"].append([r["start_ms"], r.get("end_ms") or r["start_ms"]])
        elif r["label"] == "none":
            g["none"] = True
    videos = {vid: str(paths.videos_dir() / vid) for vid, _ in grouped}
    queries = [
        {"video": vid, "query": q, "target": "any", "answers_ms": g["answers"]}
        for (vid, q), g in grouped.items()
        if g["answers"] or g["none"]
    ]
    return {"videos": videos, "queries": queries}


def precision_from_feedback(rows: list[dict[str, Any]]) -> dict[str, str]:
    """✓/✗ 를 누른 결과 중 맞음 비율 (결과 종류별)."""
    last: dict[tuple[str, str, str, int | None], str] = {}
    for r in rows:
        if r["label"] in ("correct", "wrong"):
            last[(r["video_id"], r["query"], r["kind"], r["start_ms"])] = r["label"]
    out: dict[str, str] = {}
    for kind in ("speech", "scene", "shot", "gesture"):
        labels = [v for k, v in last.items() if k[2] == kind]
        if labels:
            out[kind] = f"{sum(v == 'correct' for v in labels)}/{len(labels)} 맞음"
    return out


def _first_correct_rank(hits: list[Hit], answers: list[tuple[int, int]]) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if any(start - TOLERANCE_MS <= hit.start_ms <= end + TOLERANCE_MS for start, end in answers):
            return rank
    return None


def evaluate(spec: dict[str, Any], k: int) -> list[Outcome]:
    root = paths.PROJECT_ROOT
    searcher = Searcher()
    outcomes: list[Outcome] = []
    by_video: dict[str, list[dict[str, Any]]] = {}
    for q in spec["queries"]:
        by_video.setdefault(q["video"], []).append(q)
    for alias, queries in by_video.items():
        index = VideoIndex(resolve_index(root / spec["videos"][alias]))
        for q in queries:
            answers = [(int(s), int(e)) for s, e in q["answers_ms"]] if "answers_ms" in q else [(timecode_to_ms(s), timecode_to_ms(e)) for s, e in q["answers"]]
            hits = _hits_for(searcher.search(index, q["query"], k), q["target"])
            outcomes.append(
                Outcome(
                    video=alias,
                    query=q["query"],
                    target=q["target"],
                    negative=not answers,
                    rank=None if not answers else _first_correct_rank(hits, answers),
                    returned=len(hits),
                    top_score=hits[0].score if hits else None,
                )
            )
    return outcomes


def summarize(outcomes: list[Outcome]) -> dict[str, float | int]:
    positives = [o for o in outcomes if not o.negative]
    negatives = [o for o in outcomes if o.negative]
    ratio = lambda n, d: round(n / d, 3) if d else 0.0  # noqa: E731
    return {
        "positive_queries": len(positives),
        "hit@1": ratio(sum(o.rank == 1 for o in positives), len(positives)),
        "hit@5": ratio(sum(o.rank is not None and o.rank <= 5 for o in positives), len(positives)),
        "no_result_on_positive": ratio(sum(o.returned == 0 for o in positives), len(positives)),
        "negative_queries": len(negatives),
        "rejection_rate": ratio(sum(o.returned == 0 for o in negatives), len(negatives)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="검색 평가 (내용 미출력)")
    parser.add_argument("queries", nargs="?", default=str(DEFAULT_QUERIES))
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--feedback", action="store_true", help="검색 화면 정답 표시(eval/feedback.jsonl)로 평가")
    args = parser.parse_args()
    if args.feedback:
        rows = read_jsonl(DEFAULT_FEEDBACK)
        print(f"정답 표시 {len(rows)}건 · 결과별 맞음 비율 {precision_from_feedback(rows)}")
        spec = spec_from_feedback(rows)
    else:
        spec = read_json(Path(args.queries))
    outcomes = evaluate(spec, args.k)
    for o in outcomes:
        mark = "PASS" if o.passed else "FAIL"
        rank = "없음 질의" if o.negative else f"정답 순위 {o.rank or '-'}"
        score = f"{o.top_score:.3f}" if o.top_score is not None else "-"
        print(f"{mark} [{o.video}/{o.target}] {o.query} | {rank} | 결과 {o.returned}개 | 1위 점수 {score}")
    print(summarize(outcomes))


if __name__ == "__main__":
    main()
