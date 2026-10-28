"""vf-person — 등록 인물 관리 (실험, 내용 미출력).

  vf-person add <영상 id> <이름> --at <초>   그 시각에 화면에 있는 사람(얼굴)을 그 이름으로 등록
  vf-person assign                            모든 영상의 사람에게 등록 인물 이름을 다시 붙임
  vf-person list                              등록 인물과 영상별 매칭 수
"""

from __future__ import annotations

import argparse

import numpy as np

from videofinder.pipeline import paths
from videofinder.pipeline.faces import assign_names, load_persons, save_person
from videofinder.pipeline.jsonio import read_json, write_json


def _people_dirs() -> list:  # type: ignore[type-arg]
    return [d for d in sorted(paths.videos_dir().iterdir()) if (d / "people.json").is_file() and (d / "faces.npy").is_file()]


def assign_all() -> None:
    persons = load_persons()
    for d in _people_dirs():
        people = assign_names(read_json(d / "people.json"), np.load(d / "faces.npy"), persons)
        write_json(d / "people.json", people)
        named = [p for p in people if p["person"]]
        print(f"{d.name}: 사람 {len(people)}명 중 이름 붙음 {len(named)}명")


def add(video_id: str, name: str, at_s: float) -> None:
    d = paths.videos_dir() / video_id
    people = read_json(d / "people.json")
    faces = np.load(d / "faces.npy")
    t = int(at_s * 1000)
    candidates = [i for i, p in enumerate(people) if p["start_ms"] - 500 <= t <= p["end_ms"] + 500 and float(np.linalg.norm(faces[i])) > 0]
    if not candidates:
        raise SystemExit(f"{at_s}초에 얼굴이 잡힌 사람이 없습니다. 얼굴이 크게 보이는 다른 시각으로 다시 시도하세요.")
    best = max(candidates, key=lambda i: people[i]["faces"])  # 얼굴을 가장 많이 모은 사람
    save_person(name, faces[best])
    print(f"'{name}' 등록: 사람 #{people[best]['track_id']} (얼굴 {people[best]['faces']}회)")
    assign_all()


def main() -> None:
    parser = argparse.ArgumentParser(description="등록 인물 관리 (실험)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("video_id")
    a.add_argument("name")
    a.add_argument("--at", type=float, required=True, help="얼굴이 잘 보이는 시각(초)")
    sub.add_parser("assign")
    sub.add_parser("list")
    args = parser.parse_args()
    if args.cmd == "add":
        add(args.video_id, args.name, args.at)
    elif args.cmd == "assign":
        assign_all()
    else:
        print("등록 인물:", sorted(load_persons()))
        assign_all()


if __name__ == "__main__":
    main()
