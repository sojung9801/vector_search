"""vf-people <색인 id> — 사람 단위 분석(실험): 자세 → 추적 → 몸짓 구간을 색인에 추가한다 (내용 미출력).

출력 파일 (색인 폴더):
  people.json   사람(추적)별 등장 구간·등록 인물 이름(person)
  faces.npy     사람마다 얼굴 특징값 평균 (얼굴을 못 찾았으면 0 벡터)
  gestures.json 몸짓 구간 [{gesture, track_id, start_ms, end_ms}]
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

import numpy as np

from videofinder.pipeline import paths
from videofinder.pipeline.faces import FaceEmbedder, assign_names, face_for_head, load_persons, mean_embedding
from videofinder.pipeline.frames import iter_frames
from videofinder.pipeline.gestures import GESTURE_LABELS, track_events
from videofinder.pipeline.jsonio import write_json
from videofinder.pipeline.people import PoseDetector, link_tracks
from videofinder.retrieval.search import VideoIndex

POSE_FPS = 5
FACE_EVERY = 2  # 얼굴은 2프레임마다 (1초 2.5번)
POSE_FRAME_SIZE = 512
MIN_TRACK_MS = 1000  # 이보다 짧게 잡힌 사람은 버린다 (오검출)


def main() -> None:
    parser = argparse.ArgumentParser(description="사람 몸짓 색인 (내용 미출력)")
    parser.add_argument("video_id")
    args = parser.parse_args()
    index = VideoIndex(paths.videos_dir() / args.video_id)

    started = time.perf_counter()
    detector = PoseDetector()
    embedder = FaceEmbedder()
    frames = []
    people_frames = face_hits = 0
    for n, frame in enumerate(iter_frames(index.video_path, 1.0 / POSE_FPS, POSE_FRAME_SIZE)):
        poses = detector.detect(frame.rgb, frame.t_ms)
        if poses and n % FACE_EVERY == 0:
            faces = embedder.faces(frame.rgb)
            for pose in poses:
                head = pose.head()
                face = face_for_head(faces, head, POSE_FRAME_SIZE) if head else None
                if face is not None:
                    pose.face = face.embedding
                    face_hits += 1
        people_frames += bool(poses)
        frames.append(poses)
    pose_s = time.perf_counter() - started

    tracks = [t for t in link_tracks(frames) if t.last.t_ms - t.poses[0].t_ms >= MIN_TRACK_MS]
    events = [e for t in tracks for e in track_events(t)]
    embeddings = [mean_embedding([p.face for p in t.poses if p.face is not None]) for t in tracks]
    face_matrix = np.stack([e if e is not None else np.zeros(128, dtype=np.float32) for e in embeddings]) if tracks else np.zeros((0, 128), dtype=np.float32)
    people = [{"track_id": t.track_id, "start_ms": t.poses[0].t_ms, "end_ms": t.last.t_ms, "poses": len(t.poses), "faces": sum(p.face is not None for p in t.poses)} for t in tracks]
    np.save(index.dir / "faces.npy", face_matrix)
    write_json(index.dir / "people.json", assign_names(people, face_matrix, load_persons()))
    write_json(index.dir / "gestures.json", [e.__dict__ for e in sorted(events, key=lambda e: e.start_ms)])

    duration_s = index.meta["video"]["duration_ms"] / 1000
    print(f"프레임 {len(frames)}장 중 사람 보임 {people_frames}장 · 자세 분석 {pose_s:.0f}초 (영상 1분당 {pose_s / duration_s * 60:.0f}초)")
    print(f"사람(추적) {len(tracks)}명 (얼굴 있음 {sum(e is not None for e in embeddings)}명, 얼굴 대조 {face_hits}회) · 몸짓 구간 {len(events)}개")
    counts = Counter(e.gesture for e in events)
    for g, label in GESTURE_LABELS.items():
        times = [f"{e.start_ms // 1000}s" for e in events if e.gesture == g][:12]
        print(f"  {label}: {counts.get(g, 0)}개 {times}")


if __name__ == "__main__":
    main()
