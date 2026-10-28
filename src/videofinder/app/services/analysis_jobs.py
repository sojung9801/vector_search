"""외부 콘텐츠(content_id) 분석 작업. 한 번에 하나씩 백그라운드 스레드에서 처리한다.

입력: content_id + (url | path)
- url: mp4·webm 직접 링크는 그대로 받고, HLS(.m3u8)는 ffmpeg로 mp4에 담아 받는다. 색인 후 임시 파일은 지운다.
- path: 이 Mac의 로컬 파일 (복사하지 않음)

DRM이 걸린 재생 URL은 복호화할 수 없어 실패한다 — DRM 전 원본이나 테스트용 비DRM 영상을 넘긴다.
상태는 library/contents.json 에 저장되어 서버를 재시작해도 content_id → 색인 연결이 남는다.
"""

from __future__ import annotations

import re
import shutil
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Queue
from typing import Any, Callable
from urllib.parse import urlparse

from videofinder.pipeline import paths
from videofinder.pipeline.indexer import build_index, index_dir_for
from videofinder.pipeline.jsonio import read_json, write_json
from videofinder.pipeline.media import MediaError, run_ffmpeg, sha256_file

CONTENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
DIRECT_SUFFIXES = {".mp4", ".webm"}


class JobError(ValueError):
    pass


@dataclass
class Job:
    content_id: str
    source: str  # url 또는 로컬 경로
    status: str  # queued | downloading | indexing | done | failed
    message: str = ""
    video_id: str | None = None
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class AnalysisJobs:
    def __init__(self, on_indexed: Callable[[str], None] | None = None, store: Path | None = None) -> None:
        self._store = store or paths.library_dir() / "contents.json"
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {k: Job(**v) for k, v in read_json(self._store, {}).items()}
        # 서버가 작업 도중 꺼졌다면 끝나지 않은 작업은 실패로 둔다 (다시 요청하면 재시작)
        for job in self._jobs.values():
            if job.status not in ("done", "failed"):
                job.status, job.message = "failed", "서버가 재시작되어 중단되었습니다. 다시 분석하세요."
        self._queue: Queue[str] = Queue()
        self._on_indexed = on_indexed
        threading.Thread(target=self._worker, daemon=True, name="analysis-worker").start()

    def get(self, content_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(content_id)

    def submit(self, content_id: str, url: str | None, path: str | None) -> Job:
        if not CONTENT_ID_PATTERN.match(content_id):
            raise JobError("content_id 는 영문·숫자·_- 1~64자여야 합니다.")
        if bool(url) == bool(path):
            raise JobError("url 과 path 중 하나만 지정하세요.")
        if url and urlparse(url).scheme not in ("http", "https"):
            raise JobError("url 은 http(s) 주소여야 합니다.")
        if path and not Path(path).expanduser().is_file():
            raise JobError(f"파일이 없습니다: {path}")
        with self._lock:
            existing = self._jobs.get(content_id)
            if existing and existing.status in ("queued", "downloading", "indexing"):
                return existing
            job = Job(content_id, url or str(Path(path or "").expanduser()), "queued", "대기 중", updated_at=_now())
            self._jobs[content_id] = job
            self._save()
        self._queue.put(content_id)
        return job

    def _update(self, job: Job, status: str, message: str, video_id: str | None = None) -> None:
        with self._lock:
            job.status, job.message, job.updated_at = status, message, _now()
            if video_id:
                job.video_id = video_id
            self._save()

    def _save(self) -> None:
        write_json(self._store, {k: j.to_dict() for k, j in self._jobs.items()})

    def _worker(self) -> None:
        while True:
            content_id = self._queue.get()
            job = self.get(content_id)
            if job is None:
                continue
            download_dir = paths.work_dir() / "downloads" / content_id
            try:
                if urlparse(job.source).scheme in ("http", "https"):
                    self._update(job, "downloading", "영상 받는 중")
                    video = _download(job.source, download_dir, content_id)
                else:
                    video = Path(job.source)
                existing = index_dir_for(sha256_file(video))
                if (existing / "meta.json").is_file():
                    video_id = existing.name
                else:
                    self._update(job, "indexing", "분석 시작")
                    video_id = build_index(video, log=lambda msg: self._update(job, "indexing", msg.strip())).name
                if self._on_indexed:
                    self._on_indexed(video_id)
                self._update(job, "done", "분석 완료", video_id)
            except Exception as exc:  # noqa: BLE001 — 워커 스레드가 죽지 않게 모든 실패를 작업 상태로 남긴다
                self._update(job, "failed", f"{type(exc).__name__}: {exc}"[:500])
            finally:
                shutil.rmtree(download_dir, ignore_errors=True)


def _download(url: str, out_dir: Path, name: str) -> Path:
    """파일 이름을 content_id 로 둬서 영상 목록에서 알아볼 수 있게 한다."""
    import httpx

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix == ".m3u8":
        out = out_dir / f"{name}.mp4"
        run_ffmpeg(["-i", url, "-c", "copy", "-bsf:a", "aac_adtstoasc", str(out)], "HLS 다운로드")
        return out
    out = out_dir / f"{name}{suffix if suffix in DIRECT_SUFFIXES else '.mp4'}"
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0, connect=20.0)) as client:
        with client.stream("GET", url) as resp:
            if resp.status_code != 200:
                raise MediaError(f"영상 다운로드 실패 HTTP {resp.status_code}")
            with out.open("wb") as fh:
                for chunk in resp.iter_bytes(1024 * 1024):
                    fh.write(chunk)
    return out
