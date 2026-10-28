from videofinder.pipeline.media import MediaInfo
from videofinder.pipeline.playback import needs_playback_copy


def info(fmt: str, video: str, audio: str | None) -> MediaInfo:
    return MediaInfo("/x", 1, 1000, 1280, 720, 30.0, video, audio, 2 if audio else 0, 48000 if audio else 0, fmt)


def test_browser_safe_mp4_needs_no_copy() -> None:
    assert not needs_playback_copy(info("mov,mp4,m4a,3gp,3g2,mj2", "h264", "aac"))
    assert not needs_playback_copy(info("mov,mp4,m4a,3gp,3g2,mj2", "h264", None))


def test_webm_hevc_and_odd_audio_need_copy() -> None:
    assert needs_playback_copy(info("matroska,webm", "vp8", "vorbis"))
    assert needs_playback_copy(info("mov,mp4,m4a,3gp,3g2,mj2", "hevc", "aac"))
    assert needs_playback_copy(info("mov,mp4,m4a,3gp,3g2,mj2", "h264", "opus"))
