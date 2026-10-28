"""밀리초 ↔ 타임코드."""

from __future__ import annotations


def ms_to_timecode(ms: int, with_ms: bool = False) -> str:
    total = max(0, int(ms))
    h, rem = divmod(total, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, milli = divmod(rem, 1000)
    base = f"{h:02d}:{m:02d}:{s:02d}"
    return f"{base}.{milli:03d}" if with_ms else base


def timecode_to_ms(text: str) -> int:
    cleaned = text.strip().replace(",", ".")
    if not cleaned:
        raise ValueError("빈 타임코드")
    parts = cleaned.split(":")
    if len(parts) > 3:
        raise ValueError(f"잘못된 타임코드: {text!r}")
    try:
        seconds = float(parts[-1])
        minutes = int(parts[-2]) if len(parts) >= 2 else 0
        hours = int(parts[-3]) if len(parts) == 3 else 0
    except ValueError as exc:
        raise ValueError(f"잘못된 타임코드: {text!r}") from exc
    if min(seconds, minutes, hours) < 0:
        raise ValueError(f"음수 타임코드: {text!r}")
    return int(round((hours * 3600 + minutes * 60 + seconds) * 1000))
