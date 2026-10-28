import numpy as np

from videofinder.pipeline.asr import AsrSegment
from videofinder.pipeline.indexer import speech_chunks
from videofinder.retrieval.search import _top


def seg(start_s: float, end_s: float, text: str) -> AsrSegment:
    return AsrSegment(int(start_s * 1000), int(end_s * 1000), text, "ko", 0.0, 0.0)


def test_speech_chunks_groups_within_span() -> None:
    segments = [seg(0, 2, "a"), seg(3, 9, "b"), seg(10, 16, "c"), seg(20, 22, "d")]
    chunks = speech_chunks(segments, span_ms=15_000)
    assert [c["text"] for c in chunks] == ["a b", "b c", "c d", "d"]
    assert chunks[1]["start_ms"] == 3000 and chunks[1]["end_ms"] == 16000


def test_top_skips_nearby_hits() -> None:
    scores = np.array([0.9, 0.95, 0.5, 0.8])
    starts = [0, 2000, 60_000, 30_000]
    assert _top(scores, starts, 3) == [1, 3, 2]


def test_short_query_is_expanded_for_shot_search() -> None:
    from videofinder.retrieval.search import shot_query_text

    assert shot_query_text("여자") == "여자가 나오는 장면"
    assert shot_query_text("꽃 ") == "꽃이 나오는 장면"  # 받침 있으면 "이"
    assert shot_query_text("두 손을 든 사람") == "두 손을 든 사람"
    # 끝말이 달라도 같은 글로 찾는다
    assert shot_query_text("나비가 날아다니는 모습") == "나비가 날아다니는 장면"
    assert shot_query_text("나비가 날아다니는") == "나비가 날아다니는 장면"
    assert shot_query_text("총 쏘는 장면") == "총 쏘는 장면"
    assert shot_query_text("꼭짓점 구하는 법") == "꼭짓점 구하는 법"


def test_speech_hit_gets_previous_frame_thumb() -> None:
    from videofinder.retrieval.search import nearest_thumb

    starts = [0, 1000, 2000]
    assert nearest_thumb(starts, 1500) == "1000.jpg"
    assert nearest_thumb(starts, 1000) == "1000.jpg"
    assert nearest_thumb(starts, 0) == "0.jpg"
    assert nearest_thumb(starts, 99_000) == "2000.jpg"
    assert nearest_thumb([], 500) is None  # 프레임 없는 색인


def test_scenes_skipped_when_speech_already_answers() -> None:
    from videofinder.retrieval.search import Hit, wants_scenes

    sure = [Hit("speech", 0, 1000, 0.9)]
    low = [Hit("speech", 0, 1000, 0.02, low=True)]
    assert not wants_scenes(sure, visual=False, gesture=False)  # "꼭짓점 구하는 법": 자막이 답
    assert wants_scenes(sure, visual=True, gesture=False)  # "~장면"
    assert wants_scenes(sure, visual=False, gesture=True)  # "왼손 올림"
    assert wants_scenes(low, visual=False, gesture=False)
    assert wants_scenes([], visual=False, gesture=False)  # "여자": 자막에 없음


def test_fold_after_big_score_drop() -> None:
    from videofinder.retrieval.search import Hit, fold_after_drop

    hits = lambda *scores: [Hit("shot", i * 10000, i * 10000 + 1000, s) for i, s in enumerate(scores)]  # noqa: E731
    # "여자": 1위 0.052 → 2위 0.032 (39% 하락) 뒤는 접는다
    assert [h.low for h in fold_after_drop(hits(0.052, 0.032, 0.030, 0.025))] == [False, True, True, True]
    # "총 쏘는 장면": 고르게 내려가면(최대 24%) 그대로
    assert [h.low for h in fold_after_drop(hits(0.128, 0.113, 0.094, 0.079, 0.060))] == [False] * 5
    assert fold_after_drop(hits(0.1)) == hits(0.1)


def test_visual_query_detection() -> None:
    from videofinder.retrieval.search import is_visual_query

    assert is_visual_query("총을 쏘는 장면")
    assert is_visual_query("두 손을 든 모습 ")
    assert is_visual_query("a robot fighting scene")
    assert not is_visual_query("꼭짓점 구하는 법")
    assert not is_visual_query("장면 전환이라고 말하는 곳")


def test_drop_empty_removes_punctuation_only_segments() -> None:
    from videofinder.pipeline.indexer import drop_empty

    segments = [seg(0, 1, "."), seg(1, 2, "..."), seg(2, 3, "안녕"), seg(3, 4, " ? "), seg(4, 5, "3x")]
    assert [s.text for s in drop_empty(segments)] == ["안녕", "3x"]


def test_collapse_shots_only_for_general_queries_with_confident_speech() -> None:
    from videofinder.retrieval.search import Hit, collapse_shots

    shots = [Hit("shot", 0, 1000, 0.1)]
    confident = [Hit("speech", 0, 1000, 0.9, text="a")]
    weak = [Hit("speech", 0, 1000, 0.02, text="a", low=True)]
    assert collapse_shots(confident, shots, visual=False)[0].low
    assert not collapse_shots(confident, shots, visual=True)[0].low  # "~장면" 질의는 화면이 주 결과
    assert not collapse_shots(weak, shots, visual=False)[0].low  # 확실한 발화가 없으면 화면을 그대로 보여줌
    assert not collapse_shots([], shots, visual=False)[0].low  # 음성 없는 영상("토끼")


def test_smooth_scores_damps_single_spikes_and_keeps_runs() -> None:
    from videofinder.retrieval.search import smooth_scores

    spike = smooth_scores(np.array([0.0, 0.0, 0.9, 0.0, 0.0]), radius=1)
    run = smooth_scores(np.array([0.0, 0.6, 0.6, 0.6, 0.0]), radius=1)
    assert abs(spike[2] - 0.3) < 1e-9  # 한 장만 튄 0.9 → 0.3
    assert abs(run[2] - 0.6) < 1e-9  # 3초 이어진 0.6 → 그대로
    assert abs(smooth_scores(np.array([0.3, 0.6]), radius=1)[0] - 0.45) < 1e-9  # 끝은 있는 프레임만 평균


def test_hit_to_dict_is_json_serializable() -> None:
    import json

    from videofinder.retrieval.search import Hit

    hit = Hit("shot", 0, 1000, float(np.float32(0.1)), low=bool(np.float64(0.01) < 0.02))
    assert json.loads(json.dumps(hit.to_dict()))["low"] is True


def test_gesture_hits_filters_and_merges(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import json

    from videofinder.retrieval.search import VideoIndex, gesture_hits

    for name, data in {"meta.json": {"video": {"path": "/x.mp4", "duration_ms": 10000}}, "speech.json": [], "segments.json": [], "shots.json": []}.items():
        (tmp_path / name).write_text(json.dumps(data))
    np.save(tmp_path / "speech.npy", np.zeros((0, 768), dtype=np.float32))
    np.save(tmp_path / "shots.npy", np.zeros((0, 768), dtype=np.float32))
    (tmp_path / "gestures.json").write_text(json.dumps([
        {"gesture": "raise_left", "track_id": 0, "start_ms": 1000, "end_ms": 2000},
        {"gesture": "raise_left", "track_id": 1, "start_ms": 1500, "end_ms": 2600},
        {"gesture": "raise_right", "track_id": 0, "start_ms": 3000, "end_ms": 4000},
    ]))
    idx = VideoIndex(tmp_path)
    hits = gesture_hits(idx, "왼손 드는 장면")
    assert [(h.start_ms, h.end_ms, h.thumb) for h in hits] == [(1000, 2600, "1000.jpg")]  # 두 사람 겹침 → 하나
    assert gesture_hits(idx, "꼭짓점 구하는 법") == []
