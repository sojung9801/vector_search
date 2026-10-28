"""관절 위치 규칙으로 정해진 몸짓을 찾고, 검색어에서 몸짓을 알아본다.

거리는 어깨 너비(없으면 목~엉덩이 길이의 절반)로 나눈 상대값을 쓴다 — 사람이 화면에서 크든 작든 같은 규칙이 맞도록.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from videofinder.pipeline.people import Pose, Track

MIN_EVENT_MS = 600  # 이보다 짧게 스친 자세는 몸짓으로 보지 않는다
MAX_GAP_MS = 450  # 자세를 한 프레임 놓쳐도 같은 몸짓으로 이어 붙인다

GESTURE_LABELS = {
    "raise_left": "왼손 들기",
    "raise_right": "오른손 들기",
    "raise_both": "양손 들기",
    "hands_together": "두 손 모으기",
    "hand_on_head": "손을 머리에 (머리 긁기 등)",
    "arms_crossed": "팔짱",
    "pointing": "가리키기",
}

# 손 들기: 손목이 어깨보다 어깨너비 RAISE_LEVEL 배 아래 이상 (가슴·얼굴 높이까지 든 손).
# 사용자 표시(강의): "왼손 드는 장면" 8:21 은 손목이 어깨 아래 0.4, "양팔 번쩍" 정답 6:55·7:50·8:02 중
# 어깨 위(+0.5) 기준은 8:02 하나만 잡았다 → "번쩍"도 따로 높이 기준을 두지 않는다.
# "왼손 올림" 정답 0:25 는 손목이 어깨 아래 0.4~1.0 을 오르내림 → -0.5 는 놓치고 -0.7 은 정답 5/5, 오답 표시 0/3.
RAISE_LEVEL = -0.7
RAISE_VERB = r"(들|드는|드네|든|듦|듬|올리|올린|올라|올림|올려|올렸|번쩍)"  # "왼손 올림", "왼손 듦" 도 인식
# 두 손목이 이보다 가까우면 "들기"가 아니라 손을 모은 자세 (강의 7:21: 손목 사이 0.05~0.2, 팔꿈치는 어깨 한참 아래.
# 사용자는 "양팔 번쩍"이 아니라고 표시. 정답 7:50 은 손목 사이 3~3.8)
HANDS_TOGETHER = 0.5
# 관절이 이보다 적게 잡히거나 어깨 한쪽이 없으면 몸짓을 판단하지 않는다 (손목 하나만 잡힌 오검출이 "오른손 들기"로 잡혔다)
MIN_JOINTS = 8

# 검색어 → 몸짓. 먼저 맞는 규칙이 이긴다 (양손을 왼손·오른손보다 먼저 본다).
QUERY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("raise_both", re.compile(rf"(양손|두\s*손|양팔|두\s*팔).*{RAISE_VERB}|만세|both hands")),
    ("raise_left", re.compile(rf"왼(손|팔).*{RAISE_VERB}|left hand")),
    ("raise_right", re.compile(rf"오른(손|팔).*{RAISE_VERB}|right hand")),
    ("hand_on_head", re.compile(r"머리.*(긁|만지|손)|scratch")),
    ("arms_crossed", re.compile(r"팔짱|arms crossed")),
    ("hands_together", re.compile(r"손.*모으|손을 모|박수|합장|clap")),
    ("pointing", re.compile(r"가리키|가리킨|가리킴|손가락질|pointing")),
]
ANY_RAISE = re.compile(rf"(손|팔).*{RAISE_VERB}|raising")


@dataclass(frozen=True)
class GestureEvent:
    gesture: str
    track_id: int
    start_ms: int
    end_ms: int


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _scale(p: Pose) -> float | None:
    j = p.joints
    if "left_shoulder" in j and "right_shoulder" in j:
        return _dist(j["left_shoulder"], j["right_shoulder"])
    if "neck" in j and "root" in j:
        return _dist(j["neck"], j["root"]) / 2
    return None


def _raised(p: Pose, side: str, s: float) -> bool:
    w, sh = p.joints.get(f"{side}_wrist"), p.joints.get(f"{side}_shoulder")
    return w is not None and sh is not None and w[1] > sh[1] + RAISE_LEVEL * s  # y 는 위쪽이 큼


def _elbow_angle(p: Pose, side: str) -> float | None:
    sh, el, wr = p.joints.get(f"{side}_shoulder"), p.joints.get(f"{side}_elbow"), p.joints.get(f"{side}_wrist")
    if sh is None or el is None or wr is None:
        return None
    a = (sh[0] - el[0], sh[1] - el[1])
    b = (wr[0] - el[0], wr[1] - el[1])
    na, nb = math.hypot(*a), math.hypot(*b)
    if na == 0 or nb == 0:
        return None
    cos = max(-1.0, min(1.0, (a[0] * b[0] + a[1] * b[1]) / (na * nb)))
    return math.degrees(math.acos(cos))


def pose_gestures(p: Pose) -> set[str]:
    """한 프레임의 자세에서 성립하는 몸짓들."""
    j = p.joints
    if len(j) < MIN_JOINTS or "left_shoulder" not in j or "right_shoulder" not in j:
        return set()
    s = _scale(p)
    if s is None or s <= 0:
        return set()
    found: set[str] = set()
    left, right = _raised(p, "left", s), _raised(p, "right", s)
    together = "left_wrist" in j and "right_wrist" in j and _dist(j["left_wrist"], j["right_wrist"]) < HANDS_TOGETHER * s
    if left and right and together:
        found.add("hands_together")
    elif left and right:
        found.add("raise_both")
    elif left:
        found.add("raise_left")
    elif right:
        found.add("raise_right")
    head = p.head()
    if head is not None:
        for side in ("left", "right"):
            w, sh = j.get(f"{side}_wrist"), j.get(f"{side}_shoulder")
            if w is not None and sh is not None and w[1] > sh[1] and _dist(w, head) < 0.6 * s:
                found.add("hand_on_head")
    if all(k in j for k in ("left_wrist", "right_wrist", "left_elbow", "right_elbow")):
        if _dist(j["left_wrist"], j["right_elbow"]) < 0.5 * s and _dist(j["right_wrist"], j["left_elbow"]) < 0.5 * s:
            found.add("arms_crossed")
    for side in ("left", "right"):
        angle = _elbow_angle(p, side)
        w, sh = j.get(f"{side}_wrist"), j.get(f"{side}_shoulder")
        if angle is not None and w is not None and sh is not None and angle > 150 and abs(w[0] - sh[0]) > 1.0 * s:
            found.add("pointing")
    return found


def track_events(track: Track) -> list[GestureEvent]:
    """사람 한 명의 시간 흐름에서 MIN_EVENT_MS 이상 이어진 몸짓 구간."""
    events: list[GestureEvent] = []
    open_runs: dict[str, tuple[int, int]] = {}  # 몸짓 → (시작, 마지막으로 보인 시각)

    def close(g: str) -> None:
        start, last = open_runs.pop(g)
        if last - start >= MIN_EVENT_MS:
            events.append(GestureEvent(g, track.track_id, start, last))

    for pose in track.poses:
        now = pose_gestures(pose)
        for g in list(open_runs):
            if g not in now and pose.t_ms - open_runs[g][1] > MAX_GAP_MS:
                close(g)
        for g in now:
            start = open_runs[g][0] if g in open_runs else pose.t_ms
            open_runs[g] = (start, pose.t_ms)
    for g in list(open_runs):
        close(g)
    return sorted(events, key=lambda e: e.start_ms)


# 양손을 든 장면은 왼손·오른손을 든 장면이기도 하다 (사용자 표시: "왼손 올림" 정답 6:55·8:02 는 양손 들기)
ALSO_MATCHES = {"raise_left": {"raise_both"}, "raise_right": {"raise_both"}}


def parse_gestures(query: str) -> set[str]:
    """검색어에서 몸짓을 알아본다. 없으면 빈 집합."""
    q = query.lower()
    for gesture, pattern in QUERY_PATTERNS:
        if pattern.search(q):
            return {gesture} | ALSO_MATCHES.get(gesture, set())
    if ANY_RAISE.search(q):
        return {"raise_left", "raise_right", "raise_both"}
    return set()
