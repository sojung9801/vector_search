"""API 서버 테스트. 실제 앱을 띄우고(FastAPI TestClient) 임시 라이브러리에 색인 하나를 둔다.

색인은 빈 임베딩이라 검색 모델을 올리지 않는다 — 모델이 필요한 시험은 integration 으로 표시 (pytest -m integration).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from videofinder.app.api.server import create_app
from videofinder.app.services.video_finder import VideoFinder

VIDEO_ID = "0123456789abcdef"


def make_library(root: Path) -> Path:
    index = root / "videos" / VIDEO_ID
    (index / "thumbs").mkdir(parents=True)
    files = {
        "meta.json": {"video": {"path": str(root / "missing.mp4"), "duration_ms": 10000}, "created_at": "2026-10-26T00:00:00"},
        "segments.json": [{"start_ms": 0, "end_ms": 1500, "text": "안녕하세요", "lang": "ko"}],
        "speech.json": [],
        "shots.json": [],
    }
    for name, data in files.items():
        (index / name).write_text(json.dumps(data))
    np.save(index / "speech.npy", np.zeros((0, 768), dtype=np.float32))
    np.save(index / "shots.npy", np.zeros((0, 768), dtype=np.float32))
    Image.new("RGB", (8, 8)).save(index / "thumbs" / "0.jpg")
    return root


@pytest.fixture
def library(tmp_path: Path) -> Path:
    return make_library(tmp_path)


@pytest.fixture
def client(library: Path) -> TestClient:
    finder = VideoFinder(library_dir=library / "videos")
    app = create_app(finder, [], feedback_path=library / "feedback.jsonl", jobs_store=library / "contents.json")
    return TestClient(app)


def test_page_and_video_list(client: TestClient) -> None:
    assert "영상찾기" in client.get("/").text
    assert client.get("/api/videos").json() == [
        {"id": VIDEO_ID, "name": "missing.mp4", "duration_ms": 10000, "speech_segments": 0, "shots": 0, "created_at": "2026-10-26T00:00:00"}
    ]


def test_unknown_video_is_404(client: TestClient) -> None:
    assert client.get("/api/videos/nope").status_code == 404
    assert client.get("/api/videos/ffffffffffffffff/search", params={"q": "a"}).status_code == 404
    assert client.get("/api/videos/ffffffffffffffff/stream").status_code == 404


def test_search_rejects_bad_params(client: TestClient) -> None:
    assert client.get(f"/api/videos/{VIDEO_ID}/search", params={"q": ""}).status_code == 422
    assert client.get(f"/api/videos/{VIDEO_ID}/search", params={"q": "a", "k": 999}).status_code == 422
    assert client.get(f"/api/videos/{VIDEO_ID}/search", params={"q": "a", "k": 0}).status_code == 422


def test_thumb_serves_only_files_inside_thumbs_dir(client: TestClient) -> None:
    ok = client.get(f"/api/videos/{VIDEO_ID}/thumbs/0.jpg")
    assert ok.status_code == 200 and ok.headers["content-type"] == "image/jpeg"
    assert client.get(f"/api/videos/{VIDEO_ID}/thumbs/..%2F..%2Fmeta.json").status_code == 404
    assert client.get(f"/api/videos/{VIDEO_ID}/thumbs/missing.jpg").status_code == 404


def test_transcript_and_subtitles(client: TestClient) -> None:
    data = client.get(f"/api/videos/{VIDEO_ID}/transcript").json()
    assert len(data["segments"]) == 1 and data["scenes"] == []
    srt = client.get(f"/api/videos/{VIDEO_ID}/subtitles.srt")
    assert srt.status_code == 200 and srt.text.startswith("1\n00:00:00,000 --> 00:00:01,500")
    assert "missing.srt" in srt.headers["content-disposition"]
    assert client.get(f"/api/videos/{VIDEO_ID}/subtitles.vtt").text.startswith("WEBVTT")
    assert client.get(f"/api/videos/{VIDEO_ID}/subtitles.xyz").status_code == 422


def test_stream_without_original_file_is_410(client: TestClient) -> None:
    assert client.get(f"/api/videos/{VIDEO_ID}/stream").status_code == 410


def test_feedback_is_appended_to_file(client: TestClient, library: Path) -> None:
    body = {"video_id": VIDEO_ID, "query": "안녕", "label": "correct", "kind": "speech", "start_ms": 0, "end_ms": 1500, "score": 0.5}
    assert client.post("/api/feedback", json=body).status_code == 201
    assert client.post("/api/feedback", json={**body, "query": "  "}).status_code == 400
    assert client.post("/api/feedback", json={**body, "video_id": "ffffffffffffffff"}).status_code == 404
    assert client.post("/api/feedback", json={**body, "label": "maybe"}).status_code == 422
    rows = [json.loads(line) for line in (library / "feedback.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["query"] == "안녕" and rows[0]["at"]


def test_analysis_request_validation(client: TestClient) -> None:
    assert client.get("/api/analyses/none").status_code == 404
    assert client.post("/api/analyses", json={"content_id": "c1", "url": "http://x/a.mp4", "path": "/a.mp4"}).status_code == 400
    assert client.post("/api/analyses", json={"content_id": "bad id!", "path": "/a.mp4"}).status_code == 400


@pytest.mark.integration
def test_concurrent_searches_all_succeed(tmp_path: Path) -> None:
    """같은 모델을 두 스레드가 동시에 돌리면 검색이 멈추던 문제 (2026-10-26). 실제 라이브러리의 첫 영상으로 6개를 동시에 보낸다."""
    finder = VideoFinder()
    videos = finder.list_videos()
    if not videos:
        pytest.skip("색인된 영상이 없다 (vf-index 로 하나 추가)")
    app = create_app(finder, [], feedback_path=tmp_path / "feedback.jsonl", jobs_store=tmp_path / "contents.json")
    client = TestClient(app)
    statuses: list[int] = []

    def go() -> None:
        statuses.append(client.get(f"/api/videos/{videos[0]['id']}/search", params={"q": "그래프", "k": 8}).status_code)

    threads = [threading.Thread(target=go) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert statuses == [200] * 6
