"""평가 정답 작성용 참조 전사.

검색 시스템(Whisper turbo 배치 경로)과 다른 모델·경로를 사용한다:
- 모델: Whisper large-v3 (비-turbo)
- 30초 고정 창마다 mlx_whisper.transcribe를 따로 호출해 창별 언어를 감지
- 단어 타임스탬프 포함

이 출력은 사람이 정답 구간을 찾고 검증하는 근거로만 쓰며 색인·검색 코드에서는 읽지 않는다.

예) uv run python -m videofinder.cli.reference_transcript --video V.mp4 --out eval/reference/transcript.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from videofinder.pipeline.media import resolve_ffmpeg

REFERENCE_MODEL = "mlx-community/whisper-large-v3-mlx"
WINDOW_SECONDS = 30
SAMPLE_RATE = 16000


def load_audio(video: Path) -> np.ndarray:
    tools = resolve_ffmpeg()
    proc = subprocess.run(
        [tools.ffmpeg, "-nostdin", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="평가용 참조 전사 (Whisper large-v3, 단어 타임스탬프)")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=REFERENCE_MODEL)
    args = parser.parse_args(argv)

    import mlx_whisper

    audio = load_audio(Path(args.video))
    total = len(audio) / SAMPLE_RATE
    window = WINDOW_SECONDS * SAMPLE_RATE
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    segments: list[dict[str, object]] = []
    started = time.perf_counter()
    for idx, offset in enumerate(range(0, len(audio), window)):
        chunk = audio[offset : offset + window]
        if len(chunk) < SAMPLE_RATE:
            continue
        base = offset / SAMPLE_RATE
        result = mlx_whisper.transcribe(
            chunk,
            path_or_hf_repo=args.model,
            language=None,
            word_timestamps=True,
            condition_on_previous_text=False,
            verbose=None,
        )
        lang = str(result.get("language", ""))
        for seg in result.get("segments", []):
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            segments.append(
                {
                    "start": round(base + float(seg["start"]), 2),
                    "end": round(base + float(seg["end"]), 2),
                    "lang": lang,
                    "text": text,
                    "no_speech_prob": round(float(seg.get("no_speech_prob", 0.0)), 3),
                    "words": [
                        {"w": str(w["word"]), "s": round(base + float(w["start"]), 2), "e": round(base + float(w["end"]), 2)}
                        for w in seg.get("words", [])
                    ],
                }
            )
        if idx % 10 == 0:
            elapsed = time.perf_counter() - started
            print(f"{base / 60:6.1f}/{total / 60:.1f}분  {elapsed:6.0f}s  lang={lang}", flush=True)
            out_path.write_text(json.dumps({"model": args.model, "partial": True, "segments": segments}, ensure_ascii=False), encoding="utf-8")
    elapsed = time.perf_counter() - started
    out_path.write_text(
        json.dumps({"model": args.model, "partial": False, "elapsed_seconds": round(elapsed, 1), "audio_seconds": round(total, 1), "segments": segments}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"완료 {len(segments)} 세그먼트, {elapsed:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
