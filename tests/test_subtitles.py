from videofinder.pipeline.subtitles import to_srt, to_vtt

SEGMENTS = [{"start_ms": 1500, "end_ms": 3250, "text": " 안녕 "}, {"start_ms": 3_723_004, "end_ms": 3_725_000, "text": "끝"}]


def test_srt_numbering_and_comma_millis() -> None:
    assert to_srt(SEGMENTS) == "1\n00:00:01,500 --> 00:00:03,250\n안녕\n\n2\n01:02:03,004 --> 01:02:05,000\n끝\n"


def test_vtt_header_and_dot_millis() -> None:
    assert to_vtt(SEGMENTS).startswith("WEBVTT\n\n00:00:01.500 --> 00:00:03.250\n안녕\n")
