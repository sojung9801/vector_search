"""vf-server — 색인된 영상 목록 + 영상별 검색 HTTP API + 기본 웹 화면.

영상 추가는 vf-index <영상> 으로 색인한 뒤 목록을 새로고침한다.
Next.js 등 다른 프론트엔드에서는 이 API를 호출한다 (clients/ts/ 참고).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from videofinder.app.services.analysis_jobs import AnalysisJobs, JobError
from videofinder.app.services.video_finder import VideoFinder, VideoNotFound, video_summary
from videofinder.retrieval.search import is_gesture_query, is_visual_query
from videofinder.pipeline import paths
from videofinder.pipeline.jsonio import append_jsonl
from videofinder.pipeline.playback import PLAYBACK_FILE
from videofinder.pipeline.subtitles import to_srt, to_vtt
from videofinder.retrieval.search import VideoIndex

FRONTEND = Path(__file__).resolve().parents[1] / "frontend" / "index.html"
DEFAULT_CORS_ORIGINS = ["http://localhost:3000"]
MEDIA_TYPES = {".mp4": "video/mp4", ".webm": "video/webm"}
FEEDBACK_PATH = paths.PROJECT_ROOT / "eval" / "feedback.jsonl"


class AnalysisRequest(BaseModel):
    content_id: str
    url: str | None = None
    path: str | None = None


class FeedbackRequest(BaseModel):
    """검색 화면에서 사용자가 남기는 정답 표시. 평가(vf-eval --feedback)에 쓴다.

    correct/wrong: 특정 결과가 맞음/틀림, answer: 지금 재생 위치가 정답, none: 이 영상엔 정답 없음
    """

    video_id: str
    query: str
    label: Literal["correct", "wrong", "answer", "none"]
    kind: Literal["speech", "shot", "scene", "gesture"] | None = None
    start_ms: int | None = None
    end_ms: int | None = None
    score: float | None = None


def create_app(finder: VideoFinder, cors_origins: list[str], feedback_path: Path = FEEDBACK_PATH, jobs_store: Path | None = None) -> FastAPI:
    jobs = AnalysisJobs(on_indexed=finder.refresh, store=jobs_store)
    app = FastAPI(title="영상찾기")
    # Range 요청(영상 탐색)을 다른 출처의 <video> 에서도 쓰려면 Range/Content-Range 헤더를 열어야 한다.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Range", "Content-Type"],
        expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
    )

    def index_of(video_id: str) -> VideoIndex:
        try:
            return finder.get(video_id)
        except VideoNotFound:
            raise HTTPException(404, "영상을 찾을 수 없습니다.") from None

    @app.get("/")
    def page() -> FileResponse:
        return FileResponse(FRONTEND)

    @app.post("/api/analyses", status_code=202)
    def start_analysis(req: AnalysisRequest) -> dict[str, Any]:
        """외부 콘텐츠 분석 시작. 이미 진행 중이면 그 작업을 돌려준다."""
        try:
            return jobs.submit(req.content_id, req.url, req.path).to_dict()
        except JobError as exc:
            raise HTTPException(400, str(exc)) from None

    @app.get("/api/analyses/{content_id}")
    def analysis_status(content_id: str) -> dict[str, Any]:
        """status: queued | downloading | indexing | done | failed. done 이면 video_id 로 검색한다."""
        job = jobs.get(content_id)
        if job is None:
            raise HTTPException(404, "분석 요청이 없습니다.")
        return job.to_dict()

    @app.post("/api/feedback", status_code=201)
    def feedback(req: FeedbackRequest) -> dict[str, str]:
        index_of(req.video_id)
        if not req.query.strip():
            raise HTTPException(400, "검색어가 없는 정답 표시는 기록하지 않습니다.")
        append_jsonl(feedback_path, {**req.model_dump(), "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        return {"status": "saved"}

    @app.get("/api/videos")
    def videos() -> list[dict[str, Any]]:
        return finder.list_videos()

    @app.get("/api/videos/{video_id}")
    def video_info(video_id: str) -> dict[str, Any]:
        return video_summary(index_of(video_id))

    @app.get("/api/videos/{video_id}/search")
    def search(video_id: str, q: str = Query(min_length=1), k: int = Query(8, ge=1, le=50)) -> dict[str, Any]:
        index_of(video_id)
        started = time.perf_counter()
        result = finder.search(video_id, q, k)
        return {"video_id": video_id, "query": q, "mode": finder.mode(q), "intent": "gesture" if is_gesture_query(q) else "visual" if is_visual_query(q) else "general", "elapsed_ms": round((time.perf_counter() - started) * 1000), **{key: [h.to_dict() for h in hits] for key, hits in result.items()}}

    @app.get("/api/videos/{video_id}/transcript")
    def transcript(video_id: str) -> dict[str, Any]:
        """분석 결과 보기: 받아쓴 대사 전체와 구간 장면 설명(있을 때)."""
        index = index_of(video_id)
        return {
            "segments": [{"start_ms": s["start_ms"], "end_ms": s["end_ms"], "text": s["text"], "lang": s.get("lang")} for s in index.segments],
            "scenes": [{"start_ms": c["start_ms"], "end_ms": c["end_ms"], "text": c["text"]} for c in index.captions],
        }

    @app.get("/api/videos/{video_id}/subtitles.{fmt}")
    def subtitles(video_id: str, fmt: Literal["srt", "vtt"]) -> PlainTextResponse:
        index = index_of(video_id)
        body = to_srt(index.segments) if fmt == "srt" else to_vtt(index.segments)
        name = Path(index.video_path).stem
        return PlainTextResponse(
            body,
            media_type="application/x-subrip" if fmt == "srt" else "text/vtt",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}.{fmt}"},
        )

    @app.get("/api/videos/{video_id}/stream")
    def stream(video_id: str) -> FileResponse:
        index = index_of(video_id)
        playback = index.dir / PLAYBACK_FILE
        if playback.is_file():
            return FileResponse(playback, media_type="video/mp4")
        path = Path(index.video_path)
        if not path.is_file():
            raise HTTPException(410, "원본 영상 파일이 이동되었거나 삭제되었습니다.")
        return FileResponse(path, media_type=MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"))

    @app.get("/api/videos/{video_id}/thumbs/{name}")
    def thumb(video_id: str, name: str) -> FileResponse:
        thumbs_dir = (index_of(video_id).dir / "thumbs").resolve()
        path = (thumbs_dir / name).resolve()
        if path.parent != thumbs_dir or not path.is_file():
            raise HTTPException(404)
        return FileResponse(path)

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="영상찾기 로컬 검색 서버")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--cors-origin", action="append", help=f"API를 호출할 프론트엔드 주소 (여러 번 지정 가능, 기본 {DEFAULT_CORS_ORIGINS[0]})")
    args = parser.parse_args()
    finder = VideoFinder()
    finder.warm_up()  # 첫 검색이 모델 로딩을 기다리지 않도록 미리 올린다
    app = create_app(finder, args.cors_origin or DEFAULT_CORS_ORIGINS)
    print(f"http://127.0.0.1:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
