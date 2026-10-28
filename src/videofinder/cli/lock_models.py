"""개발자용: 채택 모델의 revision·파일·크기·해시를 configs/models.lock.json 으로 고정한다 (네트워크 필요).

앱은 이 lock 파일만 보고 설치·검증하며 런타임에 '최신' 버전을 조회하지 않는다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[3]

HF_MODELS = [
    {
        "key": "asr",
        "repo": "mlx-community/whisper-large-v3-turbo",
        "role": "음성인식 (한국어·영어·일본어, 창별 언어 감지)",
        "files": ["config.json", "weights.safetensors"],
        "license": "MIT (원본 openai/whisper-large-v3-turbo; MLX 변환본 카드에는 표기 없음)",
        "license_url": "https://huggingface.co/openai/whisper-large-v3-turbo",
        "grade": "상업 사용 가능",
        "required": True,
    },
    {
        "key": "text",
        "repo": "intfloat/multilingual-e5-base",
        "role": "발화·OCR 텍스트 임베딩 (교차언어 의미 검색)",
        "files": ["1_Pooling/config.json", "config.json", "model.safetensors", "modules.json", "sentence_bert_config.json", "sentencepiece.bpe.model", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"],
        "license": "MIT",
        "license_url": "https://huggingface.co/intfloat/multilingual-e5-base",
        "grade": "상업 사용 가능",
        "required": True,
    },
    {
        "key": "siglip",
        "repo": "google/siglip2-base-patch16-256",
        "role": "화면 검색 (다국어 질의 ↔ 프레임, 시그모이드 확률로 없는 장면 걸러냄)",
        "files": ["config.json", "model.safetensors", "preprocessor_config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"],
        "license": "Apache-2.0",
        "license_url": "https://huggingface.co/google/siglip2-base-patch16-256",
        "grade": "상업 사용 가능",
        "required": True,
    },
    {
        "key": "rerank",
        "repo": "BAAI/bge-reranker-v2-m3",
        "role": "발화 검색 재정렬 (질의·문장 쌍 판정, 관련 없는 결과 걸러냄)",
        "files": ["config.json", "model.safetensors", "sentencepiece.bpe.model", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"],
        "license": "Apache-2.0",
        "license_url": "https://huggingface.co/BAAI/bge-reranker-v2-m3",
        "grade": "상업 사용 가능",
        "required": True,
    },
]

URL_MODELS = [
    {
        "key": "face_detect",
        "role": "얼굴 찾기 (OpenCV Zoo YuNet, 사람 단위 분석 실험)",
        "url": "https://media.githubusercontent.com/media/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "filename": "face_detection_yunet_2023mar.onnx",
        "size": 232589,
        "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        "revision": "opencv_zoo-48b10fa505ae",
        "license": "MIT (opencv_zoo/models/face_detection_yunet)",
        "license_url": "https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet",
        "grade": "상업 사용 가능",
        "required": False,
    },
    {
        "key": "face_embed",
        "role": "얼굴 대조 (OpenCV Zoo SFace, 등록 인물 식별 실험 — 서비스화 전 생체정보 법무 검토 필요)",
        "url": "https://media.githubusercontent.com/media/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "filename": "face_recognition_sface_2021dec.onnx",
        "size": 38696353,
        "sha256": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        "revision": "opencv_zoo-48b10fa505ae",
        "license": "Apache-2.0 (opencv_zoo/models/face_recognition_sface)",
        "license_url": "https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface",
        "grade": "상업 사용 가능",
        "required": False,
    },
    {
        "key": "vad",
        "role": "음성 구간 감지 (Silero VAD)",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
        "filename": "silero_vad.onnx",
        "size": 643854,
        "sha256": "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6",
        "revision": "sherpa-onnx-asr-models-release",
        "license": "MIT (snakers4/silero-vad)",
        "license_url": "https://github.com/snakers4/silero-vad/blob/master/LICENSE",
        "grade": "상업 사용 가능",
        "required": True,
    },
]


def _hf_entry(api: HfApi, spec: dict[str, object]) -> dict[str, object]:
    info = api.model_info(spec["repo"], files_metadata=True)
    by_name = {s.rfilename: s for s in info.siblings}
    files = []
    for name in spec["files"]:
        s = by_name[name]
        files.append(
            {
                "path": name,
                "size": int(s.size or 0),
                "sha256": s.lfs.sha256 if s.lfs else None,
                "git_sha1": None if s.lfs else s.blob_id,
            }
        )
    return {**{k: v for k, v in spec.items() if k != "files"}, "source": "huggingface", "revision": info.sha, "files": files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", help="이 key들만 새로 고정하고 나머지 항목은 기존 lock 그대로 둔다")
    args = parser.parse_args()
    api = HfApi()
    out = ROOT / "configs" / "models.lock.json"
    if args.only:
        data = json.loads(out.read_text(encoding="utf-8"))
        kept = [e for e in data["models"] if e["key"] not in args.only]
        added = [_hf_entry(api, spec) for spec in HF_MODELS if spec["key"] in args.only]
        out.write_text(json.dumps({**data, "models": kept + added}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{out} 갱신: {[e['key'] for e in added]}")
        return
    entries = [_hf_entry(api, spec) for spec in HF_MODELS]
    for spec in URL_MODELS:
        entries.append(
            {
                **{k: v for k, v in spec.items() if k not in ("filename", "size", "sha256")},
                "source": "url",
                "files": [{"path": spec["filename"], "size": spec["size"], "sha256": spec["sha256"], "git_sha1": None, "url": spec["url"]}],
            }
        )
    out.write_text(json.dumps({"schema_version": 1, "models": entries}, ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum(f["size"] for e in entries for f in e["files"])
    print(f"{out} 작성, 모델 {len(entries)}개, 총 {total / 1e9:.2f}GB")


if __name__ == "__main__":
    main()
