"""AI 채점: 검색 결과가 질의와 맞는지 로컬 VLM(Qwen3-VL)에게 예/아니오로 묻는다.

사람이 영상을 보지 않고 정확도를 재기 위한 도구다. 답은 생성 문장이 아니라 다음 토큰의 "yes"/"no" 확률로 받는다.
같은 모델이 장면 설명에서 동작을 짐작해 쓰는 약점이 있으므로, 사람이 표시한 정답과 먼저 맞춰 보고 쓴다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from videofinder.pipeline.captioner import CAPTION_MODEL, CAPTION_REVISION, CLIP_FPS, CLIP_FRAME_SIZE, video_kwargs
from videofinder.pipeline.encoders import device
from videofinder.pipeline.frames import iter_frames

JUDGE_BEFORE_S = 1.0  # 결과 시작보다 조금 앞부터 보여준다 (화면 결과는 1초 프레임 하나라 동작 판단에 앞뒤가 필요)
JUDGE_SPAN_S = 4.0
VIDEO_QUESTION = "Does this video clip show the following? \"{query}\"\nLook carefully at what actually happens across the frames. Answer only yes or no."
TEXT_QUESTION = "Search query: \"{query}\"\nTranscript excerpt: \"{text}\"\nDoes this excerpt talk about what the query is looking for? Answer only yes or no."


class Judge:
    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.device = device()
        self.processor = AutoProcessor.from_pretrained(CAPTION_MODEL, revision=CAPTION_REVISION)
        self.model = AutoModelForImageTextToText.from_pretrained(CAPTION_MODEL, revision=CAPTION_REVISION, dtype=torch.bfloat16).to(self.device).eval()
        tok = self.processor.tokenizer
        self._yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes")]
        self._no = [tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No")]

    def clip_frames(self, video: str | Path, start_ms: int) -> list[np.ndarray]:
        start_s = max(0.0, start_ms / 1000 - JUDGE_BEFORE_S)
        frames = [f.rgb for f in iter_frames(video, 1.0 / CLIP_FPS, CLIP_FRAME_SIZE, start_s=start_s, duration_s=JUDGE_SPAN_S)]
        if len(frames) % 2:
            frames.append(frames[-1])
        return frames

    def video_yes(self, frames: list[np.ndarray], query: str) -> float:
        messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": VIDEO_QUESTION.format(query=query)}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], videos=[np.stack(frames)], videos_kwargs=video_kwargs(frames), return_tensors="pt").to(self.device)
        return self._yes_prob(inputs)

    def text_yes(self, text: str, query: str) -> float:
        messages = [{"role": "user", "content": [{"type": "text", "text": TEXT_QUESTION.format(query=query, text=text)}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self._yes_prob(self.processor(text=[prompt], return_tensors="pt").to(self.device))

    def _yes_prob(self, inputs: "object") -> float:
        with self.torch.no_grad():
            logits = self.model(**inputs).logits[0, -1].float()  # type: ignore[arg-type]
        probs = self.torch.softmax(logits, dim=-1)
        yes = float(probs[self._yes].sum())
        no = float(probs[self._no].sum())
        return yes / (yes + no) if yes + no > 0 else 0.0
