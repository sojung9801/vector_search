from videofinder.pipeline.gestures import parse_gestures, pose_gestures, track_events
from videofinder.pipeline.people import Pose, Track, link_tracks

# 정면을 보고 선 사람 (y 는 위쪽이 큼). 어깨 너비 0.2
BASE = {
    "nose": (0.5, 0.8), "left_ear": (0.55, 0.8), "right_ear": (0.45, 0.8), "neck": (0.5, 0.7),
    "left_shoulder": (0.6, 0.7), "right_shoulder": (0.4, 0.7),
    "left_elbow": (0.65, 0.55), "right_elbow": (0.35, 0.55),
    "left_wrist": (0.65, 0.4), "right_wrist": (0.35, 0.4), "root": (0.5, 0.4),
}


def pose(t: int, **moved: tuple[float, float]) -> Pose:
    return Pose(t, {**BASE, **moved})


def test_standing_has_no_gesture() -> None:
    assert pose_gestures(pose(0)) == set()


def test_raise_left_right_both() -> None:
    assert pose_gestures(pose(0, left_wrist=(0.65, 0.95))) == {"raise_left"}
    assert pose_gestures(pose(0, right_wrist=(0.35, 0.95))) == {"raise_right"}
    assert pose_gestures(pose(0, left_wrist=(0.65, 0.95), right_wrist=(0.35, 0.95))) == {"raise_both"}


def test_raise_to_chest_counts_as_raise() -> None:
    # 강의 8:21 처럼 손목이 어깨 아래 0.4 (어깨너비 0.2 → 0.08) 까지 올라옴
    assert pose_gestures(pose(0, left_wrist=(0.62, 0.62))) == {"raise_left"}
    # 강의 0:25: 어깨 아래 0.6 (0.12). 팔을 내린 손목(어깨 아래 1.5)은 들기가 아님
    assert pose_gestures(pose(0, left_wrist=(0.62, 0.58))) == {"raise_left"}


def test_hands_together_in_front_is_not_raise_both() -> None:
    # 강의 7:21: 두 손목이 어깨 높이로 올라왔지만 서로 붙어 있음 → 손 모으기
    together = pose(0, left_wrist=(0.51, 0.66), right_wrist=(0.49, 0.66))
    assert pose_gestures(together) == {"hands_together"}


def test_partial_pose_is_ignored() -> None:
    # 손목 하나만 잡힌 오검출은 몸짓으로 보지 않는다
    assert pose_gestures(Pose(0, {"right_wrist": (0.5, 0.9), "right_shoulder": (0.5, 0.7), "nose": (0.5, 0.8)})) == set()


def test_hand_on_head() -> None:
    assert "hand_on_head" in pose_gestures(pose(0, right_wrist=(0.46, 0.85), right_elbow=(0.3, 0.75)))


def test_arms_crossed() -> None:
    crossed = pose(0, left_wrist=(0.36, 0.56), right_wrist=(0.64, 0.56))
    assert "arms_crossed" in pose_gestures(crossed)


def test_pointing() -> None:
    assert "pointing" in pose_gestures(pose(0, right_elbow=(0.25, 0.7), right_wrist=(0.1, 0.7)))


def test_events_need_minimum_duration_and_bridge_gaps() -> None:
    up = {"left_wrist": (0.65, 0.62)}
    short = Track(0, [pose(0, **up), pose(200, **up), pose(400)])
    assert track_events(short) == []  # 0.2초만 든 것은 몸짓 아님
    long = Track(1, [pose(0, **up), pose(200, **up), pose(400), pose(600, **up), pose(800, **up), pose(1600)])
    events = track_events(long)
    assert [(e.gesture, e.start_ms, e.end_ms) for e in events] == [("raise_left", 0, 800)]  # 한 프레임 놓쳐도 이어 붙임


def test_link_tracks_keeps_people_apart() -> None:
    a = lambda t: Pose(t, {k: (x * 0.4, y) for k, (x, y) in BASE.items()})  # noqa: E731 왼쪽 사람
    b = lambda t: Pose(t, {k: (x * 0.4 + 0.6, y) for k, (x, y) in BASE.items()})  # noqa: E731 오른쪽 사람
    tracks = link_tracks([[a(0), b(0)], [b(200), a(200)], [a(400)]])
    assert sorted(len(t.poses) for t in tracks) == [2, 3]


def test_parse_gestures_korean_queries() -> None:
    assert parse_gestures("손흥민이 왼손 드는 장면") == {"raise_left", "raise_both"}
    assert parse_gestures("오른손을 올린 모습") == {"raise_right", "raise_both"}
    assert parse_gestures("두 손을 번쩍 든 장면") == {"raise_both"}
    assert parse_gestures("양팔 번쩍") == {"raise_both"}  # 동사 없이 "번쩍"만
    assert parse_gestures("머리를 긁는 장면") == {"hand_on_head"}
    assert parse_gestures("팔짱 낀 모습") == {"arms_crossed"}
    assert parse_gestures("칠판을 가리키는 장면") == {"pointing"}
    assert parse_gestures("박수 치는 장면") == {"hands_together"}
    for q in ("왼손 올림", "왼손을 올려", "왼손 듦", "왼손 들어 올린 모습"):
        assert parse_gestures(q) == {"raise_left", "raise_both"}, q
    assert parse_gestures("손을 드는 장면") == {"raise_left", "raise_right", "raise_both"}
    assert parse_gestures("꼭짓점 구하는 법") == set()
