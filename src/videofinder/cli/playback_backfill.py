"""vf-playback — 이미 색인된 영상 중 브라우저 재생용 사본이 필요한데 없는 것을 만든다."""

from __future__ import annotations

from pathlib import Path

from videofinder.app.services.video_finder import VIDEO_ID_PATTERN
from videofinder.pipeline import paths
from videofinder.pipeline.jsonio import read_json
from videofinder.pipeline.media import probe
from videofinder.pipeline.playback import PLAYBACK_FILE, make_playback_copy, needs_playback_copy


def main() -> None:
    for index_dir in sorted(paths.videos_dir().iterdir()):
        if not VIDEO_ID_PATTERN.match(index_dir.name) or (index_dir / PLAYBACK_FILE).is_file():
            continue
        source = Path(read_json(index_dir / "meta.json")["video"]["path"])
        if not source.is_file():
            print(f"{index_dir.name}: 원본 없음 — 건너뜀")
            continue
        if needs_playback_copy(probe(source)):
            make_playback_copy(source, index_dir / PLAYBACK_FILE)
            print(f"{index_dir.name}: 재생용 사본 생성")


if __name__ == "__main__":
    main()
