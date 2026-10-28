"""색인된 영상 한 개에 대한 자연어 검색.

발화(e5 후보 → 재정렬 확률)와 화면(SigLIP2 확률)은 따로 순위를 낸다.
발화는 재정렬 확률 SPEECH_MIN_PROB 미만을 "그런 말 없음"으로 보고 버린다.
화면 점수는 SigLIP 시그모이드 확률이며 SHOT_MIN_PROB 미만은 "그런 장면 없음"으로 보고 버린다.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import re

import numpy as np

from videofinder.pipeline.jsonio import read_json

DEDUP_MS = 5_000  # 더 높은 순위 결과와 이 간격 안이면 같은 장면으로 보고 뺀다
SHORT_QUERY_CHARS = 2  # 이하 길이 질의("야", "사과")는 발화 의미 검색 대신 원문 일치만 쓴다 (화면 검색은 shot_query_text 로 늘린다)
EXACT_LIMIT = 100
SHOT_MIN_PROB = 0.002  # 기준 강의 영상: 없는 장면(강아지·자동차) ≤ 0.0004, 실제 장면 ≥ 0.005
SHOT_CONFIDENT_PROB = 0.02  # 이 미만 화면 결과는 "관련도 낮음"으로 접는다. AI 채점(Tears of Steel): 0.002 이상 69% → 0.02 이상 81% 정답, 강의 영상은 변화 없음
# 화면 점수가 한 자리에서 이 비율 넘게 뚝 떨어지면 그 뒤는 접는다 (fold_after_drop). 사람 표시 67개(검색어 9개):
# 보이는 결과 정답률 41% → 62%, 보이는 정답 24/28 → 23/28, 접힌 오답 4/39 → 25/39. 0.25~0.35 사이는 결과가 같다.
# "여자"(정답 1곳): 1위 0.052 → 2위 0.032 (39% 하락) 뒤 7개가 접힘. "총 쏘는 장면"(정답 12곳): 최대 하락 24% 라 그대로.
SHOT_DROP = 0.3
SHOT_SMOOTH_RADIUS = 1  # 화면 점수를 앞뒤 ±1초 프레임과 평균 — 한 장만 우연히 튄 프레임이 1위가 되는 것을 줄인다
RERANK_CANDIDATES = 20  # e5 상위 몇 개를 재정렬할지 (MPS에서 20개 ≈ 0.4초)
SPEECH_MIN_PROB = 0.01  # 기준 강의 영상: 없는 말(강아지·자동차 사고) ≤ 0.004, 관련 구간 ≥ 0.012
# 이 미만(SPEECH_MIN_PROB 이상)은 "관련도 낮음"으로 표시한다. 정답 없는 점검(vf-calibrate-speech):
# 0.05 에서 있는 말 95% 통과 · 없는 말 90% 거부, 0.01 에서는 없는 말 37% 가 결과로 나옴.
# 표현을 바꾼 실제 질의는 정답도 0.01~0.05 로 나오는 경우가 있어 숨기지 않고 접어 둔다.
CONFIDENT_PROB = 0.05


@dataclass(frozen=True)
class Hit:
    kind: str  # "speech" | "shot" | "scene" | "gesture"
    start_ms: int
    end_ms: int
    score: float
    text: str | None = None
    lang: str | None = None
    thumb: str | None = None
    exact: bool = False  # 검색어가 원문에 그대로 포함됨
    low: bool = False  # 관련도 낮음 → 접어서 보여줌 (발화·장면 설명: CONFIDENT_PROB 미만, 화면: collapse_shots 참고)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VideoIndex:
    def __init__(self, index_dir: Path) -> None:
        self.dir = index_dir
        self.id = index_dir.name
        self.meta: dict[str, Any] = read_json(index_dir / "meta.json")
        self.speech: list[dict[str, Any]] = read_json(index_dir / "speech.json")
        self.segments: list[dict[str, Any]] = read_json(index_dir / "segments.json")
        self.speech_vecs = np.load(index_dir / "speech.npy")
        self.shots: list[dict[str, int]] = read_json(index_dir / "shots.json")
        self.shot_starts = [s["start_ms"] for s in self.shots]  # 오름차순. 발화 결과 썸네일 찾기용
        self.shot_vecs = np.load(index_dir / "shots.npy")
        # 구간 장면 설명 (VLM, 선택). 없으면 빈 목록
        has_captions = (index_dir / "captions.npy").is_file()
        self.captions: list[dict[str, Any]] = read_json(index_dir / "captions.json") if has_captions else []
        self.caption_vecs = np.load(index_dir / "captions.npy") if has_captions else np.zeros((0, 768), dtype=np.float32)
        # 사람 몸짓 구간 (vf-people, 실험). 없으면 빈 목록
        gestures_path = index_dir / "gestures.json"
        self.gestures: list[dict[str, Any]] = read_json(gestures_path) if gestures_path.is_file() else []
        people_path = index_dir / "people.json"
        self.people: list[dict[str, Any]] = read_json(people_path) if people_path.is_file() else []

    @property
    def video_path(self) -> str:
        return str(self.meta["video"]["path"])


def _compact(text: str) -> str:
    return "".join(text.split()).lower()


def contains(text: str, query: str) -> bool:
    """띄어쓰기·대소문자 무시 포함 여부 ("꼭지점" ↔ "꼭 지점")."""
    return _compact(query) in _compact(text)


def is_short(query: str) -> bool:
    return len(_compact(query)) <= SHORT_QUERY_CHARS


def nearest_thumb(shot_starts: list[int], ms: int) -> str | None:
    """ms 시점 직전(같은 시점 포함) 프레임의 썸네일 이름. 발화 결과를 화면 결과와 같은 그리드에 놓으려고 붙인다.
    프레임이 없는 색인이면 None."""
    if not shot_starts:
        return None
    return f"{shot_starts[max(bisect_right(shot_starts, ms) - 1, 0)]}.jpg"


# "총을 쏘는 장면"처럼 화면을 찾는 질의. 발화 검색은 '그 말이 나온 곳'을 찾으므로 실제 장면과 시점이 다를 수 있다.
VISUAL_QUERY = re.compile(r"(장면|모습|화면|scene|shot|footage|clip)\s*$", re.IGNORECASE)
SHOT_SUFFIX = re.compile(r"\s*(장면|모습|화면)\s*$")  # 화면 검색 글에서 "장면"으로 통일하는 끝말


def is_visual_query(query: str) -> bool:
    return bool(VISUAL_QUERY.search(query.strip()))


def smooth_scores(scores: np.ndarray, radius: int = SHOT_SMOOTH_RADIUS) -> np.ndarray:
    """1초 간격 프레임 점수를 앞뒤 radius 장과 평균 (양 끝은 있는 프레임만). 진짜 장면은 몇 초 이어지므로 유지되고,
    한 장만 튄 점수는 내려간다."""
    if radius <= 0 or len(scores) == 0:
        return scores
    kernel = np.ones(2 * radius + 1)
    window = slice(radius, radius + len(scores))  # mode="same" 은 프레임 수가 kernel 보다 적으면 길이가 어긋난다
    total = np.convolve(scores, kernel, mode="full")[window]
    count = np.convolve(np.ones(len(scores)), kernel, mode="full")[window]
    return total / count


def collapse_shots(speech: list[Hit], shots: list[Hit], visual: bool) -> list[Hit]:
    """말을 찾는 질의("꼭짓점 구하는 법")에 확실한 발화 결과가 있으면 화면 결과는 접어 둔다.

    AI 채점(강의 영상): 이런 질의의 화면 결과는 정답률이 낮고, 화면 점수 기준값을 0.002~0.2 로 바꿔도
    정답률이 55~60% 로 그대로라 기준값으로는 거를 수 없다. 발화가 없으면("토끼") 화면이 주 결과라 그대로 둔다.
    """
    if visual or not any(not h.low for h in speech):
        return shots
    return [replace(h, low=True) for h in shots]


def wants_scenes(speech: list[Hit], visual: bool, gesture: bool) -> bool:
    """장면 설명을 재정렬할지. 말을 찾는 질의에 확실한 발화가 있으면 건너뛴다 — 설명 20개 재정렬에 약 1.3초
    (설명 글이 발화의 3배 길이). "~장면/모습"·몸짓 질의와 자막에 없는 질의("여자", "칠판을 지우는 모습")는 본다."""
    return visual or gesture or all(h.low for h in speech)


def fold_after_drop(shots: list[Hit], min_drop: float = SHOT_DROP) -> list[Hit]:
    """점수순 화면 결과에서 가장 큰 상대 하락이 min_drop 이상이면 그 뒤를 모두 접는다 (정답이 한두 곳뿐인 질의)."""
    if len(shots) < 2:
        return shots
    drops = [(a.score - b.score) / a.score if a.score > 0 else 0.0 for a, b in zip(shots, shots[1:])]
    biggest = max(drops)
    if biggest < min_drop:
        return shots
    cut = drops.index(biggest) + 1
    return shots[:cut] + [replace(h, low=True) for h in shots[cut:]]


def shot_query_text(query: str) -> str:
    """화면(SigLIP) 검색에 넣을 글. 뜻이 같아도 끝말에 따라 점수가 크게 달라서 "~ 장면" 꼴로 맞춘다.

    - 짧은 질의("여자") → "여자가 나오는 장면": SigLIP 은 캡션 문장으로 학습돼 낱말 하나에는 점수가 거의 0 이다.
      강의 "여자": 정답 4:25 가 0.0008 로 6위(기준값 미만) → 0.052 로 1위. BBB 토끼·풀밭·하늘·꽃: 기준값 넘는 결과 5/25 → 12/25.
    - "~모습"·"~화면" → "~장면", "~는"(관형형) → "~는 장면": BBB "나비가 날아다니는 모습" 0.001(결과 없음) → 0.057,
      Tears "총을 쏘는 모습" 0.015 → 0.102. AI 채점 48개 질의: 1위 정답률 73% 그대로, 확실한(0.02 이상) 정답 74 → 88개.
    뜻 없는 "야"·"음"도 0.04 대로 올라오지만, 낱말 그대로 써도 0.006~0.02 라 점수로는 어차피 못 거른다.
    """
    q = query.strip()
    if is_short(q):
        last = q[-1]
        particle = "이" if "가" <= last <= "힣" and (ord(last) - 0xAC00) % 28 else "가"
        return f"{q}{particle} 나오는 장면"
    if SHOT_SUFFIX.search(q):
        return SHOT_SUFFIX.sub(" 장면", q).strip()
    if q.endswith("는"):
        return q + " 장면"
    return q


def _top(scores: np.ndarray, starts: list[int], k: int) -> list[int]:
    picked: list[int] = []
    for i in np.argsort(-scores):
        if all(abs(starts[i] - starts[j]) >= DEDUP_MS for j in picked):
            picked.append(int(i))
            if len(picked) == k:
                break
    return picked


def is_gesture_query(query: str) -> bool:
    from videofinder.pipeline.gestures import parse_gestures

    return bool(parse_gestures(query))


def gesture_hits(idx: VideoIndex, query: str) -> list[Hit]:
    """검색어에 정해진 몸짓("왼손 드는")이 있으면 관절 규칙으로 찾은 구간을 시간순으로 돌려준다 (점수 없음).

    검색어에 등록 인물 이름("선생님이 왼손…")이 있으면 그 사람의 몸짓만. 여러 사람의 같은 몸짓이 겹치면 하나로 합친다.
    """
    from videofinder.pipeline.faces import load_persons
    from videofinder.pipeline.gestures import GESTURE_LABELS, parse_gestures

    wanted = parse_gestures(query)
    if not wanted or not idx.gestures:
        return []
    who = {p["track_id"]: p.get("person") for p in idx.people}
    # 등록 인물 전체와 비교한다 — 이 영상에 없는 인물을 물으면 결과가 없어야 한다 (이름 조건을 무시하면 안 됨)
    names = {n for n in load_persons() if n in query}
    events = [e for e in idx.gestures if e["gesture"] in wanted and (not names or who.get(e["track_id"]) in names)]
    hits: list[Hit] = []
    for e in sorted(events, key=lambda e: e["start_ms"]):
        if hits and e["start_ms"] <= hits[-1].end_ms:
            last = hits[-1]
            hits[-1] = replace(last, end_ms=max(last.end_ms, e["end_ms"]))
            continue
        thumb = f"{(e['start_ms'] // 1000) * 1000}.jpg"
        person = who.get(e["track_id"])
        label = f"{person} · {GESTURE_LABELS[e['gesture']]}" if person else GESTURE_LABELS[e["gesture"]]
        hits.append(Hit("gesture", e["start_ms"], e["end_ms"], 1.0, text=label, thumb=thumb))
    return hits[:EXACT_LIMIT]


class Searcher:
    """모델을 한 번 올려 두고 여러 영상 색인에 재사용한다. 질의 쪽만 쓰므로 fp16 + SigLIP 글 모델만 올린다."""

    def __init__(self) -> None:
        from videofinder.pipeline.encoders import Reranker, SiglipEncoder, TextEncoder

        self.text = TextEncoder(half=True)
        self.reranker = Reranker(half=True)
        self.siglip = SiglipEncoder(text_only=True, half=True)

    def search(self, idx: VideoIndex, query: str, k: int = 10) -> dict[str, list[Hit]]:
        short = is_short(query)
        visual = is_visual_query(query)  # 화면을 찾는 질의면 발화 결과는 모두 접어 둔다
        query_vec = self.text.query(query) if len(idx.speech) or len(idx.captions) else None
        speech = (
            self._exact_segments(idx, query)
            if short
            else [
                Hit("speech", r["start_ms"], r["end_ms"], p, text=r["text"], lang=r["lang"], thumb=nearest_thumb(idx.shot_starts, r["start_ms"]), exact=contains(r["text"], query), low=visual or p < CONFIDENT_PROB)
                for r, p in self._rerank(idx.speech, idx.speech_vecs, query, query_vec, k)
            ]
        )
        gesture = is_gesture_query(query)
        scenes = (
            [
                Hit("scene", r["start_ms"], r["end_ms"], p, text=r["text"], thumb=f"{r['start_ms']}.jpg", low=p < CONFIDENT_PROB)
                for r, p in self._rerank(idx.captions, idx.caption_vecs, query, query_vec, k)
            ]
            if wants_scenes(speech, visual, gesture)
            else []
        )
        shots: list[Hit] = []
        if len(idx.shots):
            scores = smooth_scores(self.siglip.probabilities(idx.shot_vecs @ self.siglip.query(shot_query_text(query))))
            for i in _top(scores, idx.shot_starts, k):
                if scores[i] < SHOT_MIN_PROB:
                    break
                s = idx.shots[i]
                shots.append(Hit("shot", s["start_ms"], s["end_ms"], float(scores[i]), thumb=f"{s['start_ms']}.jpg", low=bool(scores[i] < SHOT_CONFIDENT_PROB)))
        gestures = gesture_hits(idx, query)
        # 짧은 질의("여자")는 자막에 그 낱말이 있어도 화면을 찾는 뜻일 수 있어 화면 결과를 접지 않는다
        shots = collapse_shots(speech, fold_after_drop(shots), visual or short)
        # 몸짓 질의라도 화면 결과는 접지 않는다: 사용자 표시("왼손 올림")에서 화면 결과 8개 중 5개가 정답이었다
        return {"speech": speech, "shots": shots, "scenes": scenes, "gestures": gestures}

    def _rerank(self, records: list[dict[str, Any]], vecs: np.ndarray, query: str, query_vec: np.ndarray | None, k: int) -> list[tuple[dict[str, Any], float]]:
        """e5 후보 → 재정렬 확률 → SPEECH_MIN_PROB 미만 제거. 발화와 장면 설명에 같이 쓴다."""
        if not records or query_vec is None:
            return []
        starts = [r["start_ms"] for r in records]
        candidates = _top(vecs @ query_vec, starts, RERANK_CANDIDATES)
        probs = self.reranker.probabilities(query, [records[i]["text"] for i in candidates])
        out: list[tuple[dict[str, Any], float]] = []
        for j in _top(probs, [starts[i] for i in candidates], k):
            if probs[j] < SPEECH_MIN_PROB:
                break
            out.append((records[candidates[j]], float(probs[j])))
        return out

    @staticmethod
    def _exact_segments(idx: VideoIndex, query: str) -> list[Hit]:
        """짧은 질의: 원문 세그먼트에서 그대로 찾아 시간순으로 돌려준다 (순위 개념 없음 → k 대신 상한만)."""
        matched = [s for s in idx.segments if contains(s["text"], query)]
        return [Hit("speech", s["start_ms"], s["end_ms"], 1.0, text=s["text"], lang=s["lang"], thumb=nearest_thumb(idx.shot_starts, s["start_ms"]), exact=True) for s in matched[:EXACT_LIMIT]]
