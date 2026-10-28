"""Silero VAD (sherpa-onnx)로 음성 구간을 찾고 Whisper 30초 창으로 묶는다."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


@dataclass(frozen=True)
class SpeechRegion:
    start_s: float
    end_s: float


def detect_speech(audio: np.ndarray, model_file: Path, threshold: float = 0.5, min_silence_s: float = 0.35, min_speech_s: float = 0.25, max_speech_s: float = 20.0) -> list[SpeechRegion]:
    import sherpa_onnx

    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = str(model_file)
    cfg.silero_vad.threshold = threshold
    cfg.silero_vad.min_silence_duration = min_silence_s
    cfg.silero_vad.min_speech_duration = min_speech_s
    cfg.silero_vad.max_speech_duration = max_speech_s
    cfg.silero_vad.window_size = 512
    cfg.sample_rate = SAMPLE_RATE
    cfg.num_threads = 2
    vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=max(60, int(len(audio) / SAMPLE_RATE) + 10))
    regions: list[SpeechRegion] = []

    def drain() -> None:
        while not vad.empty():
            seg = vad.front
            start = seg.start / SAMPLE_RATE
            regions.append(SpeechRegion(start, start + len(seg.samples) / SAMPLE_RATE))
            vad.pop()

    step = SAMPLE_RATE * 10
    for i in range(0, len(audio), step):
        vad.accept_waveform(audio[i : i + step])
        drain()
    vad.flush()
    drain()
    return regions


def build_windows(regions: list[SpeechRegion], total_s: float, max_window_s: float = 28.0, max_gap_s: float = 2.5, pad_s: float = 0.25) -> list[tuple[float, float]]:
    """음성 구간을 Whisper 입력 창(≤ max_window_s)으로 묶는다. 긴 무음에서는 창을 끊는다."""
    if max_window_s <= 0 or max_window_s > 30:
        raise ValueError("max_window_s는 0~30초")
    pieces: list[tuple[float, float]] = []
    for r in regions:
        s, e = max(0.0, r.start_s - pad_s), min(total_s, r.end_s + pad_s)
        while e - s > max_window_s:
            pieces.append((s, s + max_window_s))
            s += max_window_s
        if e > s:
            pieces.append((s, e))
    windows: list[tuple[float, float]] = []
    for s, e in pieces:
        if windows:
            ws, we = windows[-1]
            if s - we <= max_gap_s and e - ws <= max_window_s:
                windows[-1] = (ws, max(we, e))
                continue
            if s < we:  # 패딩으로 겹치면 앞 창 끝으로 당김
                s = we
                if e - s <= 0.05:
                    continue
        windows.append((s, e))
    return windows
