"""임베딩 모델 로더. 설치된 고정 revision 모델만 로컬 경로로 연다 (네트워크 없음).

- 발화: multilingual-e5-base (질의 "query: ", 문서 "passage: " 접두어)로 후보를 뽑고 bge-reranker-v2-m3로 재정렬.
  e5 코사인은 정답·오답 차이가 0.03 남짓이라 "관련 없음"을 판정할 수 없다.
- 화면: SigLIP2 (다국어 질의를 직접 이해, 시그모이드 확률이라 "그런 장면 없음"을 판단할 수 있다)
  다국어 CLIP(clip-ViT-B-32-multilingual-v1)은 한국어 질의에서 엉뚱한 장면을 1위로 올리고 없는 장면도 비슷한 점수를 줘서 교체.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from videofinder.pipeline.models import model_path

if TYPE_CHECKING:
    from PIL.Image import Image
    from sentence_transformers import CrossEncoder, SentenceTransformer


RERANK_BATCH = 16  # 32 로 한 번에 처리해도 빨라지지 않았다 (vf-bench-search)


def device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def _dtype_kwargs(half: bool) -> dict[str, Any]:
    """half: 검색 서버용 반정밀도(fp16). 처음부터 그 형식으로 올린다 (fp32 로 올린 뒤 바꾸면 그 순간 메모리가 치솟는다).

    vf-bench-search 측정: 검색어 20개의 1위·결과 목록 동일, 점수 차 ≤ 0.003, 상주 메모리 6.1GB → 3.4GB.
    색인(저장되는 벡터)은 fp32 로 만든다.
    """
    import torch

    return {"dtype": torch.float16} if half else {}


def _load(key: str, half: bool) -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(str(model_path(key)), device=device(), local_files_only=True, model_kwargs=_dtype_kwargs(half))


class TextEncoder:
    def __init__(self, half: bool = False) -> None:
        self.model = _load("text", half)

    def passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self._encode([f"passage: {t}" for t in texts], batch_size)

    def query(self, text: str) -> np.ndarray:
        return self._encode([f"query: {text}"], 1)[0]

    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        out = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(out, dtype=np.float32)


class Reranker:
    def __init__(self, half: bool = False) -> None:
        from sentence_transformers import CrossEncoder

        self.model: "CrossEncoder" = CrossEncoder(str(model_path("rerank")), device=device(), max_length=256, local_files_only=True, model_kwargs=_dtype_kwargs(half))

    def probabilities(self, query: str, texts: list[str]) -> np.ndarray:
        """질의와 각 문장이 관련 있을 확률 (모델 기본 시그모이드 출력)."""
        out = self.model.predict([(query, t) for t in texts], batch_size=RERANK_BATCH, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(out, dtype=np.float32)


class SiglipEncoder:
    def __init__(self, text_only: bool = False, half: bool = False) -> None:
        """text_only: 검색 서버는 질의(글)만 인코딩하므로 이미지 쪽 모델을 버려 메모리를 줄인다."""
        import torch
        from transformers import AutoModel, AutoProcessor

        path = str(model_path("siglip"))
        self.torch = torch
        self.device = device()
        self.model = AutoModel.from_pretrained(path, local_files_only=True, dtype=torch.float16 if half else torch.float32).eval()
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.logit_scale = float(self.model.logit_scale.exp().item())
        self.logit_bias = float(self.model.logit_bias.item())
        if text_only:
            del self.model.vision_model
        self.model.to(self.device)  # 이미지 쪽을 버린 뒤 GPU 로 옮겨 GPU 메모리를 아낀다

    def _normalized(self, features: "Any") -> np.ndarray:
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        return self.torch.nn.functional.normalize(features.float(), dim=-1).cpu().numpy()

    def images(self, images: list["Image"]) -> np.ndarray:
        with self.torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            return self._normalized(self.model.get_image_features(**inputs))

    def query(self, text: str) -> np.ndarray:
        # SigLIP는 학습 때처럼 max_length 패딩을 해야 점수가 맞는다.
        with self.torch.no_grad():
            inputs = self.processor(text=[text], padding="max_length", max_length=64, return_tensors="pt").to(self.device)
            return self._normalized(self.model.get_text_features(**inputs))[0]

    def probabilities(self, cosines: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(cosines * self.logit_scale + self.logit_bias)))
