"""얼굴 찾기·대조 (실험). OpenCV Zoo YuNet(MIT) + SFace(Apache-2.0).

사람 단위 분석에서 추적한 사람마다 얼굴 특징값(128차원)을 모아 평균을 내고,
등록한 인물(library/persons.json)과 코사인 유사도로 대조해 이름을 붙인다.
얼굴로 사람을 식별하는 것은 생체정보 처리라 서비스화하려면 법무 검토가 필요하다 (지금은 로컬 실험).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from videofinder.pipeline import paths
from videofinder.pipeline.jsonio import read_json, write_json
from videofinder.pipeline.models import model_path

SFACE_MATCH = 0.363  # SFace 권장 코사인 기준값 (OpenCV 문서)
MIN_FACE_SCORE = 0.8
PERSONS_FILE = "persons.json"


@dataclass(frozen=True)
class Face:
    box: tuple[float, float, float, float]  # x, y, w, h (픽셀, 위쪽이 0)
    embedding: np.ndarray  # 정규화된 128차원


class FaceEmbedder:
    def __init__(self) -> None:
        import cv2

        self.cv2 = cv2
        self.detector = cv2.FaceDetectorYN.create(str(model_path("face_detect") / "face_detection_yunet_2023mar.onnx"), "", (320, 320), MIN_FACE_SCORE)
        self.recognizer = cv2.FaceRecognizerSF.create(str(model_path("face_embed") / "face_recognition_sface_2021dec.onnx"), "")

    def faces(self, rgb: np.ndarray) -> list[Face]:
        bgr = self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        self.detector.setInputSize((w, h))
        _, found = self.detector.detect(bgr)
        out: list[Face] = []
        for row in found if found is not None else []:
            aligned = self.recognizer.alignCrop(bgr, row)
            vec = self.recognizer.feature(aligned).flatten().astype(np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                out.append(Face((float(row[0]), float(row[1]), float(row[2]), float(row[3])), vec / norm))
        return out


def face_for_head(faces: list[Face], head_xy: tuple[float, float], size: int) -> Face | None:
    """자세의 머리 위치(0~1, y 위쪽이 1)를 품은 얼굴 박스. 여러 개면 가장 큰 것."""
    px, py = head_xy[0] * size, (1 - head_xy[1]) * size
    inside = [f for f in faces if f.box[0] <= px <= f.box[0] + f.box[2] and f.box[1] <= py <= f.box[1] + f.box[3]]
    return max(inside, key=lambda f: f.box[2] * f.box[3]) if inside else None


def mean_embedding(vectors: list[np.ndarray]) -> np.ndarray | None:
    if not vectors:
        return None
    mean = np.mean(np.stack(vectors), axis=0)
    norm = float(np.linalg.norm(mean))
    return (mean / norm).astype(np.float32) if norm > 0 else None


# ---------- 등록 인물 ----------

def load_persons() -> dict[str, np.ndarray]:
    data: dict[str, list[float]] = read_json(paths.library_dir() / PERSONS_FILE, {})
    return {name: np.asarray(vec, dtype=np.float32) for name, vec in data.items()}


def save_person(name: str, embedding: np.ndarray) -> None:
    """같은 이름이 있으면 기존 특징값과 평균 낸다 (여러 장면으로 등록할수록 안정적)."""
    persons = load_persons()
    merged = mean_embedding([persons[name], embedding]) if name in persons else embedding
    persons[name] = merged if merged is not None else embedding
    write_json(paths.library_dir() / PERSONS_FILE, {n: [round(float(x), 6) for x in v] for n, v in persons.items()})


def match_person(embedding: np.ndarray | None, persons: dict[str, np.ndarray]) -> tuple[str | None, float]:
    if embedding is None or not persons:
        return None, 0.0
    name, score = max(((n, float(v @ embedding)) for n, v in persons.items()), key=lambda x: x[1])
    return (name, score) if score >= SFACE_MATCH else (None, score)


def assign_names(people: list[dict[str, Any]], embeddings: np.ndarray, persons: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """people.json 의 사람마다 등록 인물 이름을 붙인다 (얼굴이 없거나 닮지 않으면 None)."""
    out = []
    for i, p in enumerate(people):
        vec = embeddings[i] if i < len(embeddings) and float(np.linalg.norm(embeddings[i])) > 0 else None
        name, score = match_person(vec, persons)
        out.append({**p, "person": name, "person_score": round(score, 3)})
    return out
