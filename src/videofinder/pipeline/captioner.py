"""짧은 구간 장면 설명 생성 (VLM). 프레임 한 장으로는 알 수 없는 동작("떨어진다", "달린다")을 문장으로 남긴다.

원본에서 CLIP_FPS 로 촘촘히 뽑은 프레임을 CLIP_S 초씩 묶어 영상으로 넣는다 (1초 1장으로는 사과가 떨어지는 것 같은 빠른 동작을 놓침).
설명은 영어로 받은 뒤 같은 모델로 한국어 번역을 붙인다 — 재정렬 모델이 언어가 다른 쌍(한국어 질의 ↔ 영어 설명)에 약하다.
(한 번에 두 언어로 쓰라고 하면 2B 모델이 형식을 무시하고 영어만 쓴다.)
설명은 e5 로 임베딩해 발화와 같은 방식(후보 → 재정렬)으로 검색한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from videofinder.pipeline.encoders import device
from videofinder.pipeline.frames import Frame, iter_frames

CAPTION_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
CAPTION_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
CLIP_S = 4
CLIP_FPS = 4
CLIP_FRAME_SIZE = 384
# 작은 VLM 은 물체에 흔히 붙는 동작을 짐작해 쓰는 경향이 있다 (보이기만 하는 것을 "떨어진다", "난다"로).
# 프레임 사이에서 실제로 보이는 움직임만 쓰고, 가만히 있으면 그렇다고 쓰게 한다.
PROMPT = (
    "In one sentence, describe this short video clip. Mention the visible characters and objects. "
    "Describe only motion that you can clearly see change across the frames; "
    "if a character or object stays still, say that it is still. "
    "Do not assume an action just because it is typical for that object or character."
)
TRANSLATE_PROMPT = "Translate into natural Korean. Output only the Korean sentence.\n\n"


def video_kwargs(frames: list[np.ndarray], fps: float = CLIP_FPS) -> dict[str, Any]:
    """미리 뽑은 프레임을 Qwen3-VL 에 넘길 때의 옵션.

    - video_metadata: 없으면 모델이 24fps 로 가정해 4초 구간을 0.7초로 이해한다 (동작 판단이 틀어짐).
    - cap_pixels_per_frame: 끄면 짧은 구간도 프레임당 거의 원본 해상도 토큰을 써서 느리고 메모리를 많이 쓴다.
    """
    from transformers.video_utils import VideoMetadata

    n = len(frames)
    height, width = frames[0].shape[:2]
    metadata = VideoMetadata(total_num_frames=n, fps=float(fps), width=width, height=height, duration=n / fps, frames_indices=list(range(n)))
    return {"video_metadata": [metadata], "do_sample_frames": False, "cap_pixels_per_frame": True}


class ClipCaptioner:
    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.device = device()
        self.processor = AutoProcessor.from_pretrained(CAPTION_MODEL, revision=CAPTION_REVISION)
        self.model = AutoModelForImageTextToText.from_pretrained(CAPTION_MODEL, revision=CAPTION_REVISION, dtype=torch.bfloat16).to(self.device).eval()

    def caption(self, frames: list[np.ndarray]) -> str:
        """frames: (H, W, 3) uint8, CLIP_FPS 간격. 반환: "영어 설명\n한국어 번역"."""
        english = self._describe(frames)
        return f"{english}\n{self._translate(english)}"

    def _translate(self, english: str) -> str:
        messages = [{"role": "user", "content": [{"type": "text", "text": TRANSLATE_PROMPT + english}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)
        return self._generate(inputs, 96)

    def _generate(self, inputs: "Any", max_new_tokens: int) -> str:
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = out[:, inputs["input_ids"].shape[1] :]
        return str(self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]).strip()

    def _describe(self, frames: list[np.ndarray]) -> str:
        messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": PROMPT}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if len(frames) % 2:  # Qwen3-VL 은 2프레임씩 묶어 처리하므로 짝수로 맞춘다 (마지막 구간)
            frames = [*frames, frames[-1]]
        video = np.stack(frames)
        inputs = self.processor(
            text=[text],
            videos=[video],
            videos_kwargs=video_kwargs(frames),
            return_tensors="pt",
        ).to(self.device)
        return self._generate(inputs, 64)


MIN_TAIL_S = 2  # 마지막 구간이 이보다 짧으면 앞 구간에 붙인다 (움직임을 판단할 프레임이 너무 적음)


def iter_clips(video: str | Path, clip_s: int = CLIP_S, fps: int = CLIP_FPS, size: int = CLIP_FRAME_SIZE) -> Iterator[tuple[int, list[np.ndarray]]]:
    """(구간 시작 ms, 프레임들). 원본을 한 번 순서대로 읽으며 clip_s 초씩 끊는다."""
    return group_clips(iter_frames(video, 1.0 / fps, size), clip_s, fps)


def group_clips(frames: Iterable[Frame], clip_s: int, fps: int) -> Iterator[tuple[int, list[np.ndarray]]]:
    per_clip = clip_s * fps
    pending: tuple[int, list[np.ndarray]] | None = None  # 짧은 꼬리를 붙일 수 있게 한 구간 늦게 내보낸다
    batch: list[np.ndarray] = []
    start_ms = 0
    for frame in frames:
        if not batch:
            start_ms = frame.t_ms
        batch.append(frame.rgb)
        if len(batch) == per_clip:
            if pending:
                yield pending
            pending, batch = (start_ms, batch), []
    if batch and pending and len(batch) < MIN_TAIL_S * fps:
        pending = (pending[0], pending[1] + batch)
        batch = []
    if pending:
        yield pending
    if batch:
        yield start_ms, batch
