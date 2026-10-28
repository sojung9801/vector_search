"""영상찾기 진입점. 다른 코드에서는 이 클래스만 import 해서 쓴다.

    from videofinder.app.services.video_finder import VideoFinder

    finder = VideoFinder()                       # 가볍다. 검색 모델은 첫 검색(또는 warm_up) 때 올린다 (수 초)
    video_id = finder.add("/path/video.mp4")     # 색인이 없으면 만든다
    finder.list_videos()                         # 색인된 영상 목록
    result = finder.search(video_id, "두 손을 번쩍 든 장면")

검색은 항상 영상 한 개 안에서만 한다 (여러 영상 통합 검색 없음).
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

from videofinder.pipeline import paths
from videofinder.pipeline.indexer import Log, build_index, index_dir_for
from videofinder.pipeline.media import sha256_file
from videofinder.retrieval.search import Hit, Searcher, VideoIndex, is_short

VIDEO_ID_PATTERN = re.compile(r"^[0-9a-f]{16}$")


class VideoNotFound(LookupError):
    pass


def video_summary(index: VideoIndex) -> dict[str, Any]:
    video = index.meta["video"]
    return {
        "id": index.id,
        "name": Path(video["path"]).name,
        "duration_ms": video["duration_ms"],
        "speech_segments": len(index.speech),
        "shots": len(index.shots),
        "created_at": index.meta.get("created_at"),
    }


class VideoFinder:
    def __init__(self, library_dir: Path | None = None) -> None:
        self.library_dir = library_dir or paths.videos_dir()
        self._searcher: Searcher | None = None
        self._indexes: dict[str, VideoIndex] = {}
        # 서버는 요청마다 다른 스레드에서 search 를 부른다. 같은 모델을 두 스레드가 동시에 돌리면
        # 검색이 멈추거나(2개) 프로세스가 죽는다(6개) — 검색은 한 번에 하나만.
        self._search_lock = threading.Lock()

    def warm_up(self) -> None:
        """검색 모델을 미리 올린다 (수 초). 안 부르면 첫 검색 때 올린다."""
        with self._search_lock:
            self._load_searcher()

    def _load_searcher(self) -> Searcher:
        # _search_lock 을 잡고 부른다
        if self._searcher is None:
            self._searcher = Searcher()
        return self._searcher

    def list_videos(self) -> list[dict[str, Any]]:
        if not self.library_dir.is_dir():
            return []
        ids = sorted(d.name for d in self.library_dir.iterdir() if VIDEO_ID_PATTERN.match(d.name) and (d / "meta.json").is_file())
        return sorted((video_summary(self.get(i)) for i in ids), key=lambda v: str(v["name"]))

    def get(self, video_id: str) -> VideoIndex:
        if not VIDEO_ID_PATTERN.match(video_id):
            raise VideoNotFound(video_id)
        if video_id not in self._indexes:
            index_dir = self.library_dir / video_id
            if not (index_dir / "meta.json").is_file():
                raise VideoNotFound(video_id)
            self._indexes[video_id] = VideoIndex(index_dir)
        return self._indexes[video_id]

    def add(self, video: str | Path, log: Log = print) -> str:
        """영상을 색인하고 id를 돌려준다. 같은 내용(SHA-256)의 색인이 있으면 재사용."""
        index_dir = resolve_index(video, log)
        self.refresh(index_dir.name)
        return index_dir.name

    def refresh(self, video_id: str) -> None:
        """색인이 새로 만들어졌을 때 캐시된 이전 색인을 버린다."""
        self._indexes.pop(video_id, None)

    def search(self, video_id: str, query: str, k: int = 8) -> dict[str, list[Hit]]:
        """{"speech": [...], "shots": [...]}. 관련 없는 결과는 빠지므로 빈 목록일 수 있다."""
        index = self.get(video_id)
        with self._search_lock:
            return self._load_searcher().search(index, query, k)

    @staticmethod
    def mode(query: str) -> str:
        """"exact" = 짧은 검색어라 자막 원문 일치만 찾음, "semantic" = 의미 검색."""
        return "exact" if is_short(query) else "semantic"


def resolve_index(target: str | Path, log: Log = print) -> Path:
    path = Path(target).expanduser()
    if path.is_dir():
        return path
    existing = index_dir_for(sha256_file(path))
    if (existing / "meta.json").is_file():
        return existing
    return build_index(path, log=log)
