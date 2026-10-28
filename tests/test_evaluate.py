from videofinder.cli.evaluate import Outcome, _first_correct_rank, summarize
from videofinder.retrieval.search import Hit


def hit(start_s: float) -> Hit:
    return Hit("shot", int(start_s * 1000), int(start_s * 1000) + 1000, 0.5)


def test_first_correct_rank_uses_tolerance() -> None:
    hits = [hit(10), hit(50), hit(103)]
    assert _first_correct_rank(hits, [(100_000, 101_000)]) == 3  # 103초는 끝+3초 안
    assert _first_correct_rank(hits, [(200_000, 210_000)]) is None


def test_summarize_counts_positive_and_negative() -> None:
    outcomes = [
        Outcome("v", "a", "shots", negative=False, rank=1, returned=5, top_score=0.9),
        Outcome("v", "b", "shots", negative=False, rank=4, returned=5, top_score=0.4),
        Outcome("v", "c", "shots", negative=False, rank=None, returned=0, top_score=None),
        Outcome("v", "d", "speech", negative=True, rank=None, returned=0, top_score=None),
        Outcome("v", "e", "speech", negative=True, rank=None, returned=2, top_score=0.2),
    ]
    s = summarize(outcomes)
    assert s["hit@1"] == 0.333 and s["hit@5"] == 0.667 and s["no_result_on_positive"] == 0.333
    assert s["rejection_rate"] == 0.5


def test_spec_from_feedback_groups_answers_and_negatives() -> None:
    from videofinder.cli.evaluate import precision_from_feedback, spec_from_feedback

    rows = [
        {"video_id": "v1", "query": "a", "label": "answer", "kind": None, "start_ms": 20000, "end_ms": 20000},
        {"video_id": "v1", "query": "a", "label": "correct", "kind": "scene", "start_ms": 4000, "end_ms": 8000},
        {"video_id": "v1", "query": "a", "label": "wrong", "kind": "shot", "start_ms": 9000, "end_ms": 10000},
        {"video_id": "v1", "query": "b", "label": "none", "kind": None, "start_ms": None, "end_ms": None},
        {"video_id": "v1", "query": "c", "label": "correct", "kind": "shot", "start_ms": 1000, "end_ms": 2000},
        {"video_id": "v1", "query": "c", "label": "wrong", "kind": "shot", "start_ms": 1000, "end_ms": 2000},
    ]
    spec = spec_from_feedback(rows)
    by_query = {q["query"]: q["answers_ms"] for q in spec["queries"]}
    assert sorted(by_query["a"]) == [[4000, 8000], [20000, 20000]]
    assert by_query["b"] == []
    assert "c" not in by_query  # 마지막 표시가 wrong → 정답 없음, none 표시도 없음 → 평가 제외
    assert precision_from_feedback(rows) == {"scene": "1/1 맞음", "shot": "0/2 맞음"}
