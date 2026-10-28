"""받아쓴 대사(segments.json)를 자막 파일(SRT·VTT)로 만든다."""

from __future__ import annotations

from typing import Any

from videofinder.pipeline.timecode import ms_to_timecode


def _stamp(ms: int, decimal: str) -> str:
    return ms_to_timecode(ms, with_ms=True).replace(".", decimal)


def to_srt(segments: list[dict[str, Any]]) -> str:
    blocks = [f"{n}\n{_stamp(s['start_ms'], ',')} --> {_stamp(s['end_ms'], ',')}\n{str(s['text']).strip()}\n" for n, s in enumerate(segments, start=1)]
    return "\n".join(blocks)


def to_vtt(segments: list[dict[str, Any]]) -> str:
    blocks = [f"{_stamp(s['start_ms'], '.')} --> {_stamp(s['end_ms'], '.')}\n{str(s['text']).strip()}\n" for s in segments]
    return "WEBVTT\n\n" + "\n".join(blocks)
