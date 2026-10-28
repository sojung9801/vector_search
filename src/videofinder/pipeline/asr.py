"""배치 Whisper 음성인식 (MLX).

- VAD로 만든 30초 이하 창들을 묶어 인코더를 배치 실행한다.
- 창마다 언어를 감지하되 허용 언어(ko/en/ja) 안에서만 고른다 → 한·영·일 혼합 영상 대응.
- 언어별로 디코딩 배치를 나눠 타임스탬프 토큰으로 문장 구간을 복원한다.
- 반복/저확신 창은 온도 상승 재시도(openai/whisper fallback 규칙), 그래도 나쁘면 버린다.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

SAMPLE_RATE = 16000
TIME_PRECISION = 0.02


@dataclass(frozen=True)
class AsrSegment:
    start_ms: int
    end_ms: int
    text: str
    lang: str
    avg_logprob: float
    no_speech_prob: float

    def to_dict(self) -> dict[str, Any]:
        return {"start_ms": self.start_ms, "end_ms": self.end_ms, "text": self.text, "lang": self.lang, "avg_logprob": round(self.avg_logprob, 3), "no_speech_prob": round(self.no_speech_prob, 3)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AsrSegment":
        return cls(int(d["start_ms"]), int(d["end_ms"]), str(d["text"]), str(d.get("lang", "")), float(d.get("avg_logprob", 0)), float(d.get("no_speech_prob", 0)))


@dataclass
class AsrStats:
    windows: int = 0
    fallbacks: int = 0
    dropped: int = 0
    seconds: float = 0.0
    languages: dict[str, int] = field(default_factory=dict)


def parse_timestamped_tokens(tokens: list[int], timestamp_begin: int, decode: Callable[[list[int]], str], window_len_s: float) -> list[tuple[float, float, str]]:
    """Whisper 출력 토큰 → [(start_s, end_s, text)] (창 기준 상대 시간)."""
    out: list[tuple[float, float, str]] = []
    cur_start: float | None = None
    cur: list[int] = []
    for t in tokens:
        if t >= timestamp_begin:
            ts = (t - timestamp_begin) * TIME_PRECISION
            if cur and cur_start is not None:
                text = decode(cur).strip()
                if text:
                    out.append((cur_start, max(ts, cur_start), text))
                cur = []
                cur_start = None
            else:
                cur_start = ts
        else:
            if cur_start is None:
                cur_start = out[-1][1] if out else 0.0
            cur.append(t)
    if cur:
        text = decode(cur).strip()
        if text:
            start = cur_start if cur_start is not None else 0.0
            out.append((start, max(window_len_s, start), text))
    return out


class WhisperBatchASR:
    def __init__(self, model_dir: Path, languages: tuple[str, ...] = ("ko", "en", "ja"), batch_size: int = 8) -> None:
        import mlx.core as mx
        from mlx_whisper.load_models import load_model

        self.mx = mx
        self.model = load_model(str(model_dir), dtype=mx.float16)
        self.languages = tuple(languages)
        self.batch_size = max(1, batch_size)
        from mlx_whisper.tokenizer import get_tokenizer

        self._get_tokenizer = get_tokenizer
        base = get_tokenizer(self.model.is_multilingual, num_languages=self.model.num_languages, language="en", task="transcribe")
        self.timestamp_begin = base.timestamp_begin
        self._lang_token_ids = {lang: base.to_language_token(lang) for lang in self.languages}

    def _mel(self, audio: np.ndarray) -> "Any":
        from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim

        mel = log_mel_spectrogram(audio, n_mels=self.model.dims.n_mels, padding=max(0, N_SAMPLES - len(audio)))
        return pad_or_trim(mel, N_FRAMES, axis=-2)

    def _restricted_languages(self, features: "Any") -> list[str]:
        mx = self.mx
        from mlx_whisper.decoding import detect_language  # noqa: F401  (형식 확인용)

        tokenizer = self._get_tokenizer(self.model.is_multilingual, num_languages=self.model.num_languages, language="en", task="transcribe")
        n = features.shape[0]
        x = mx.array([[tokenizer.sot]] * n)
        logits = self.model.logits(x, features)[:, 0]
        ids = mx.array(list(self._lang_token_ids.values()))
        sub = mx.take(logits, ids, axis=-1)
        best = mx.argmax(sub, axis=-1).tolist()
        return [self.languages[i] for i in best]

    def _decode(self, features: "Any", lang: str, temperature: float) -> list[Any]:
        from mlx_whisper.decoding import DecodingOptions, DecodingTask

        opts = DecodingOptions(task="transcribe", language=lang, temperature=temperature, fp16=True, without_timestamps=False)
        return DecodingTask(self.model, opts).run(features)

    def transcribe(self, audio: np.ndarray, windows: list[tuple[float, float]], offset_s: float = 0.0, progress: Callable[[int, int], None] | None = None) -> tuple[list[AsrSegment], AsrStats]:
        mx = self.mx
        stats = AsrStats(windows=len(windows))
        started = time.perf_counter()
        segments: list[AsrSegment] = []
        for b in range(0, len(windows), self.batch_size):
            batch = windows[b : b + self.batch_size]
            mels = [self._mel(audio[int(s * SAMPLE_RATE) : int(e * SAMPLE_RATE)]) for s, e in batch]
            features = self.model.encoder(mx.stack(mels).astype(mx.float16))
            mx.eval(features)
            langs = self._restricted_languages(features)
            for lang in sorted(set(langs)):
                idx = [i for i, l in enumerate(langs) if l == lang]
                feats = features[mx.array(idx)] if len(idx) != len(langs) else features
                results = self._decode(feats, lang, 0.0)
                for i, res in zip(idx, results):
                    res = self._maybe_fallback(features[i : i + 1], lang, res, stats)
                    if res is None:
                        stats.dropped += 1
                        continue
                    ws, we = batch[i]
                    tokenizer = self._get_tokenizer(self.model.is_multilingual, num_languages=self.model.num_languages, language=lang, task="transcribe")
                    for s, e, text in parse_timestamped_tokens(list(res.tokens), self.timestamp_begin, tokenizer.decode, we - ws):
                        start = offset_s + ws + min(s, we - ws)
                        end = offset_s + ws + min(e, we - ws)
                        if end <= start:
                            end = start + 0.2
                        segments.append(AsrSegment(int(round(start * 1000)), int(round(end * 1000)), text, lang, float(res.avg_logprob), float(res.no_speech_prob)))
                    stats.languages[lang] = stats.languages.get(lang, 0) + 1
            if progress:
                progress(min(b + self.batch_size, len(windows)), len(windows))
        stats.seconds = time.perf_counter() - started
        segments.sort(key=lambda s: (s.start_ms, s.end_ms))
        return segments, stats

    def _maybe_fallback(self, feature: "Any", lang: str, res: Any, stats: AsrStats) -> Any | None:
        def bad(r: Any) -> bool:
            return r.compression_ratio > 2.4 or r.avg_logprob < -1.0

        def silent(r: Any) -> bool:
            return r.no_speech_prob > 0.6 and r.avg_logprob < -1.0

        if silent(res):
            return None
        if not bad(res):
            return res
        for temp in (0.2, 0.4, 0.6):
            stats.fallbacks += 1
            alt = self._decode(feature, lang, temp)[0]
            if silent(alt):
                return None
            if not bad(alt):
                return alt
        return None if res.compression_ratio > 2.4 else res

    def close(self) -> None:
        self.model = None
        gc.collect()
        self.mx.clear_cache()
