"""단일 영상 색인: 발화(ASR → e5) + 화면(프레임 → SigLIP2 샷).

출력: library/videos/<sha256 앞 16자>/
  meta.json            영상 정보·처리 시간
  segments.json        ASR 원본 세그먼트
  speech.json/.npy     발화 검색 구간(15초 묶음)과 e5 임베딩
  shots.json/.npy      샘플링한 프레임별 구간과 SigLIP2 임베딩
  thumbs/<ms>.jpg      프레임 썸네일
  playback.mp4         브라우저 재생용 사본 (원본이 webm 등일 때만)
"""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from videofinder.pipeline import paths
from videofinder.pipeline.asr import AsrSegment, WhisperBatchASR
from videofinder.pipeline.frames import Frame, iter_frames
from videofinder.pipeline.jsonio import write_json
from videofinder.pipeline.media import MediaInfo, read_audio_pcm, sha256_file, validate_video
from videofinder.pipeline.models import model_path
from videofinder.pipeline.playback import PLAYBACK_FILE, make_playback_copy, needs_playback_copy
from videofinder.pipeline.vad import build_windows, detect_speech

FRAME_BATCH = 32
CHUNK_SPAN_MS = 15_000  # 발화 검색 단위 길이
HAS_WORD = re.compile(r"\w")

Log = Callable[[str], None]


def index_dir_for(sha256: str) -> Path:
    return paths.videos_dir() / sha256[:16]


def build_index(video: str | Path, frame_interval_s: float = 1.0, log: Log = print) -> Path:
    info = validate_video(video)
    sha = sha256_file(info.path)
    out = index_dir_for(sha)
    tmp = out.with_name(out.name + ".building")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    timings: dict[str, float] = {}

    started = time.perf_counter()
    segments = drop_empty(_transcribe(info, log)) if info.has_audio else []
    timings["asr_s"] = time.perf_counter() - started
    _write_speech(tmp, segments, log)
    timings["speech_embed_s"] = time.perf_counter() - started - timings["asr_s"]

    mark = time.perf_counter()
    shots = _index_frames(info, tmp, frame_interval_s, log)
    timings["frames_s"] = time.perf_counter() - mark

    if needs_playback_copy(info):
        mark = time.perf_counter()
        log("재생용 사본 만드는 중")
        make_playback_copy(info.path, tmp / PLAYBACK_FILE)
        timings["playback_s"] = time.perf_counter() - mark

    write_json(
        tmp / "meta.json",
        {
            "video": info.to_dict(),
            "sha256": sha,
            "frame_interval_s": frame_interval_s,
            "speech_segments": len(segments),
            "shots": shots,
            "timings": {k: round(v, 1) for k, v in timings.items()},
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    log(f"색인 완료: {out} (발화 {len(segments)}개, 샷 {shots}개, {time.perf_counter() - started:.0f}초)")
    return out


def _transcribe(info: MediaInfo, log: Log) -> list[AsrSegment]:
    total_s = info.duration_ms / 1000
    audio = read_audio_pcm(info.path, 0.0, total_s)
    windows = build_windows(detect_speech(audio, model_path("vad") / "silero_vad.onnx"), total_s)
    log(f"음성인식: 창 {len(windows)}개")
    asr = WhisperBatchASR(model_path("asr"))
    try:
        segments, stats = asr.transcribe(audio, windows, progress=lambda d, n: log(f"  음성인식 {d}/{n}"))
    finally:
        asr.close()
    log(f"음성인식 완료: {len(segments)}개 세그먼트, {stats.seconds:.0f}초, 언어 {stats.languages}")
    return segments


def drop_empty(segments: list[AsrSegment]) -> list[AsrSegment]:
    """글자·숫자가 없는 세그먼트를 버린다. Whisper 가 효과음 구간에 "." 만 남기는 경우가 있고,
    재정렬 모델은 빈 문장에 엉뚱하게 높은 확률(0.18)을 줘서 검색 1위로 올라온다."""
    return [s for s in segments if HAS_WORD.search(s.text)]


def speech_chunks(segments: list[AsrSegment], span_ms: int = CHUNK_SPAN_MS) -> list[dict[str, Any]]:
    """세그먼트마다 그 시점부터 span_ms 안의 세그먼트를 이어 붙인 구간을 만든다 (겹침 허용, 검색 시 중복 제거).

    Whisper 세그먼트는 "이렇게", "꼭지점." 처럼 짧아 단독 임베딩은 의미가 약하다.
    """
    chunks: list[dict[str, Any]] = []
    for i, first in enumerate(segments):
        group = [first]
        for seg in segments[i + 1 :]:
            if seg.end_ms - first.start_ms > span_ms:
                break
            group.append(seg)
        chunks.append({"start_ms": first.start_ms, "end_ms": group[-1].end_ms, "text": " ".join(s.text for s in group), "lang": first.lang})
    return chunks


def _write_speech(out: Path, segments: list[AsrSegment], log: Log) -> None:
    from videofinder.pipeline.encoders import TextEncoder

    write_json(out / "segments.json", [s.to_dict() for s in segments])
    chunks = speech_chunks(segments)
    write_json(out / "speech.json", chunks)
    if not chunks:
        np.save(out / "speech.npy", np.zeros((0, 768), dtype=np.float32))
        return
    np.save(out / "speech.npy", TextEncoder().passages([c["text"] for c in chunks]))
    log(f"발화 임베딩 완료: {len(chunks)}개")


def _index_frames(info: MediaInfo, out: Path, interval_s: float, log: Log) -> int:
    """프레임마다 임베딩을 그대로 저장한다. 비슷한 프레임을 묶어 평균 내면 손을 드는 몇 초짜리 순간이 묻힌다."""
    from PIL import Image

    from videofinder.pipeline.encoders import SiglipEncoder

    encoder = SiglipEncoder()
    thumbs = out / "thumbs"
    thumbs.mkdir()
    step_ms = int(round(interval_s * 1000))
    shots: list[dict[str, int]] = []
    vecs: list[np.ndarray] = []

    def flush(batch: list[Frame]) -> None:
        images = [Image.fromarray(f.rgb) for f in batch]
        vecs.append(encoder.images(images))
        for frame, image in zip(batch, images):
            image.save(thumbs / f"{frame.t_ms}.jpg", quality=80)
            shots.append({"start_ms": frame.t_ms, "end_ms": min(frame.t_ms + step_ms, info.duration_ms)})
        log(f"  프레임 {len(shots)}개")

    batch: list[Frame] = []
    for frame in iter_frames(info.path, interval_s):
        batch.append(frame)
        if len(batch) == FRAME_BATCH:
            flush(batch)
            batch = []
    if batch:
        flush(batch)

    write_json(out / "shots.json", shots)
    np.save(out / "shots.npy", np.concatenate(vecs).astype(np.float32) if vecs else np.zeros((0, 768), dtype=np.float32))
    return len(shots)
