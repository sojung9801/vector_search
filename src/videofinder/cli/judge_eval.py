"""vf-judge — AI 채점으로 검색 정확도 측정 (영상 내용은 출력하지 않는다).

  vf-judge --validate        사람이 표시한 ✓/✗(eval/feedback.jsonl)와 채점 AI 판정이 얼마나 맞는지
  vf-judge [queries.json]    질의 목록을 검색하고 결과마다 채점 → 결과 종류별 정확도, 점수 기준값별 정밀도

출력: 시각·점수·채점 확률·통계뿐. 자막·설명 문장·프레임은 출력하지 않는다.

메모리: 검색 모델(약 5GB)과 채점 AI(약 4.3GB)를 동시에 올리면 16GB Mac 에서 스왑이 난다.
그래서 ① 검색만 모두 끝내고 검색 모델을 내린 뒤 ② 채점 AI 만 올려 판정한다.
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from videofinder.app.services.video_finder import VideoFinder
from videofinder.pipeline import paths
from videofinder.pipeline.jsonio import read_json, read_jsonl, write_json
from videofinder.cli.judge import Judge
from videofinder.retrieval.search import Hit, VideoIndex

JUDGE_YES = 0.5
DEFAULT_QUERIES = paths.PROJECT_ROOT / "eval" / "judge_queries.json"
FEEDBACK = paths.PROJECT_ROOT / "eval" / "feedback.jsonl"
RESULTS = paths.PROJECT_ROOT / "eval" / "judge_results.json"
THRESHOLDS = {"shot": [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2], "speech": [0.01, 0.02, 0.05, 0.1, 0.2, 0.5], "scene": [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]}


@dataclass(frozen=True)
class Candidate:
    """① 단계 검색 결과. text 는 발화 채점용으로 메모리에만 두고 파일에 쓰지 않는다."""

    video: str
    query: str
    kind: str
    rank: int
    start_ms: int
    score: float
    text: str | None


@dataclass
class Judged:
    video: str
    query: str
    kind: str
    rank: int
    start_ms: int
    score: float
    judge: float

    @property
    def ok(self) -> bool:
        return self.judge >= JUDGE_YES


def judge_hit(judge: Judge, index: VideoIndex, query: str, kind: str, start_ms: int, text: str | None) -> float:
    if kind == "speech":
        return judge.text_yes(text or "", query)
    return judge.video_yes(judge.clip_frames(index.video_path, start_ms), query)


def _index(video_id: str) -> VideoIndex:
    return VideoIndex(paths.videos_dir() / video_id)


def validate(judge: Judge) -> None:
    last: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for r in read_jsonl(FEEDBACK):
        if r["label"] in ("correct", "wrong") and str(r["query"]).strip():
            last[(r["video_id"], r["query"], r["kind"], r["start_ms"])] = r
    agree = 0
    for (video_id, query, kind, start_ms), r in sorted(last.items(), key=lambda kv: kv[0][3]):
        index = _index(video_id)
        text = next((s["text"] for s in index.speech if s["start_ms"] == start_ms), None) if kind == "speech" else None
        p = judge_hit(judge, index, query, kind, start_ms, text)
        human = r["label"] == "correct"
        agree += human == (p >= JUDGE_YES)
        print(f"{kind:6s} {start_ms // 1000:5d}s 사람 {'✓' if human else '✗'} | 채점 예 확률 {p:.3f} → {'✓' if p >= JUDGE_YES else '✗'}")
    print(f"\n일치 {agree}/{len(last)}")


def search_all(spec: dict[str, Any], k: int) -> list[Candidate]:
    """① 검색 모델만 올려 모든 질의를 검색한다."""
    finder = VideoFinder()
    out: list[Candidate] = []
    for item in spec["queries"]:
        result: dict[str, list[Hit]] = finder.search(item["video"], item["query"], k)
        for kind_key, kind in (("speech", "speech"), ("scenes", "scene"), ("shots", "shot")):
            for rank, h in enumerate(result.get(kind_key, []), 1):
                out.append(Candidate(item["video"], item["query"], kind, rank, h.start_ms, h.score, h.text))
    print(f"① 검색 완료: 질의 {len(spec['queries'])}개, 채점할 결과 {len(out)}개")
    return out


def free_search_models() -> None:
    import torch

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def judge_all(judge: Judge, spec: dict[str, Any], candidates: list[Candidate]) -> list[Judged]:
    """② 채점 AI 만 올려 판정한다."""
    rows: list[Judged] = []
    indexes: dict[str, VideoIndex] = {}
    for item in spec["queries"]:
        video_id, query = item["video"], item["query"]
        index = indexes.setdefault(video_id, _index(video_id))
        line = [f"[{video_id[:6]}] {query}"]
        for kind in ("speech", "scene", "shot"):
            mine = [c for c in candidates if c.video == video_id and c.query == query and c.kind == kind]
            if not mine:
                continue
            judged = [Judged(c.video, c.query, c.kind, c.rank, c.start_ms, c.score, judge_hit(judge, index, query, kind, c.start_ms, c.text)) for c in mine]
            rows.extend(judged)
            line.append(f"{kind} {sum(j.ok for j in judged)}/{len(judged)}" + (" (1위 ✓)" if judged[0].ok else " (1위 ✗)"))
        print(" | ".join(line) if len(line) > 1 else f"{line[0]} | 결과 없음", flush=True)
    return rows


def report(rows: list[Judged]) -> None:
    print("\n결과 종류 | 질의 수 | 1위 정답률 | 전체 정답률(정밀도)")
    for kind in ("speech", "scene", "shot"):
        ks = [r for r in rows if r.kind == kind]
        if not ks:
            continue
        firsts = [r for r in ks if r.rank == 1]
        print(f"{kind:6s} | {len(firsts):3d} | {np.mean([r.ok for r in firsts]):.0%} | {np.mean([r.ok for r in ks]):.0%} ({sum(r.ok for r in ks)}/{len(ks)})")
    print("\n점수 기준값별: 남는 결과 수 / 그중 정답률")
    for kind, ts in THRESHOLDS.items():
        ks = [r for r in rows if r.kind == kind]
        if not ks:
            continue
        cells = []
        for t in ts:
            kept = [r for r in ks if r.score >= t]
            cells.append(f"{t:g}: {len(kept)}개 {np.mean([r.ok for r in kept]):.0%}" if kept else f"{t:g}: 0개")
        print(f"{kind:6s} | " + " | ".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser(description="AI 채점 정확도 (내용 미출력)")
    parser.add_argument("queries", nargs="?", default=str(DEFAULT_QUERIES))
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--video", help="이 영상 id(앞부분만 써도 됨)의 질의만 돌린다 — 한 번에 조금씩 나눠 돌릴 때")
    args = parser.parse_args()
    if args.validate:
        validate(Judge())
        return
    spec = read_json(Path(args.queries))
    if args.video:
        spec = {**spec, "queries": [q for q in spec["queries"] if q["video"].startswith(args.video)]}
    candidates = search_all(spec, args.k)
    free_search_models()
    rows = judge_all(Judge(), spec, candidates)
    write_json(RESULTS, [r.__dict__ for r in rows])
    report(rows)


if __name__ == "__main__":
    main()
