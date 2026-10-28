"""사람 단위 분석 (실험): 프레임마다 사람 몸 자세(관절)를 찾고, 같은 사람을 이어 붙이고, 정해진 몸짓을 규칙으로 판단한다.

- 자세: Apple Vision VNDetectHumanBodyPoseRequest (macOS 내장, 무료). 관절 좌표는 정사각형 레터박스 프레임 기준 0~1,
  y 는 위쪽이 1. 왼쪽·오른쪽은 그 사람 기준이다 (카메라를 보고 있으면 화면에서는 반대쪽에 보임).
- 추적: 이전 프레임 사람과 몸 박스가 가장 많이 겹치면 같은 사람으로 본다 (단순 IoU).
- 몸짓: 관절 위치 규칙 (gestures.py). 화면 전체를 요약하는 SigLIP 과 달리 작은 몸짓도 사람 기준으로 잡는다.

멀리 찍혀 사람이 작으면 관절을 못 찾는다.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np

JOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "neck",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "root", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
)
MIN_JOINT_CONF = 0.3
TRACK_IOU = 0.3  # 이전 프레임 사람과 박스가 이만큼 겹치면 같은 사람
TRACK_GAP_MS = 1500  # 이 시간 넘게 안 보이면 다른 사람으로 본다


@dataclass
class Pose:
    t_ms: int
    joints: dict[str, tuple[float, float]]  # 이름 → (x, y), 신뢰도 MIN_JOINT_CONF 이상만
    face: np.ndarray | None = None  # 이 프레임에서 머리 위치에 맞는 얼굴 특징값 (faces.py)

    def head(self) -> tuple[float, float] | None:
        pts = [self.joints[k] for k in ("nose", "left_eye", "right_eye", "left_ear", "right_ear") if k in self.joints]
        if not pts:
            return None
        return sum(x for x, _ in pts) / len(pts), sum(y for _, y in pts) / len(pts)

    def box(self) -> tuple[float, float, float, float]:
        xs = [p[0] for p in self.joints.values()]
        ys = [p[1] for p in self.joints.values()]
        return min(xs), min(ys), max(xs), max(ys)


@dataclass
class Track:
    track_id: int
    poses: list[Pose] = field(default_factory=list)

    @property
    def last(self) -> Pose:
        return self.poses[-1]


def _vision_joint_keys() -> dict[str, object]:
    import Vision

    keys: dict[str, object] = {}
    for name in JOINTS:
        camel = "".join(w.capitalize() for w in name.split("_"))
        const = getattr(Vision, f"VNHumanBodyPoseObservationJointName{camel}", None)
        if const is not None:
            keys[name] = const
    return keys


class PoseDetector:
    def __init__(self) -> None:
        self._keys = _vision_joint_keys()

    def detect(self, rgb: np.ndarray, t_ms: int) -> list[Pose]:
        import Foundation
        import Vision
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=90)
        data = Foundation.NSData.dataWithBytes_length_(buf.getvalue(), len(buf.getvalue()))
        handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
        request = Vision.VNDetectHumanBodyPoseRequest.alloc().init()
        ok, _ = handler.performRequests_error_([request], None)
        if not ok:
            return []
        poses: list[Pose] = []
        for obs in request.results() or []:
            joints: dict[str, tuple[float, float]] = {}
            for name, key in self._keys.items():
                point, _ = obs.recognizedPointForJointName_error_(key, None)
                if point is not None and point.confidence() >= MIN_JOINT_CONF:
                    loc = point.location()
                    joints[name] = (float(loc.x), float(loc.y))
            if len(joints) >= 4:
                poses.append(Pose(t_ms, joints))
        return poses


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def link_tracks(frames: list[list[Pose]]) -> list[Track]:
    """프레임별 사람 목록 → 사람별 시간 흐름. 박스가 가장 많이 겹치는 최근 사람에 이어 붙인다."""
    tracks: list[Track] = []
    for poses in frames:
        used: set[int] = set()
        for pose in sorted(poses, key=lambda p: -(p.box()[2] - p.box()[0]) * (p.box()[3] - p.box()[1])):
            best, best_iou = None, TRACK_IOU
            for tr in tracks:
                if tr.track_id in used or pose.t_ms - tr.last.t_ms > TRACK_GAP_MS:
                    continue
                score = _iou(tr.last.box(), pose.box())
                if score >= best_iou:
                    best, best_iou = tr, score
            if best is None:
                best = Track(len(tracks))
                tracks.append(best)
            best.poses.append(pose)
            used.add(best.track_id)
    return tracks
