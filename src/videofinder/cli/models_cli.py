"""CLI: 모델 설치 상태 확인·설치 (앱 설정 화면과 같은 코드).

예) uv run vf-models list
    uv run vf-models install
"""

from __future__ import annotations

import argparse
import sys

from videofinder.pipeline import models


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="모델 설치 관리")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    inst = sub.add_parser("install")
    inst.add_argument("keys", nargs="*")
    inst.add_argument("--offline", action="store_true", help="로컬 HF 캐시에서만 복사")
    args = parser.parse_args(argv)
    if args.cmd == "list":
        for m in models.status():
            mark = "설치됨" if m["installed"] else ("캐시 있음" if m["cached_elsewhere"] == m["size"] else "미설치")
            print(f"{m['key']:<11} {mark:<6} {m['size'] / 1e6:8.0f}MB  {m['grade']:<8} {m['name']}  rev={m['revision'][:12]}")
        return 0
    last = [0.0]

    def progress(p: dict) -> None:
        pct = p["done_bytes"] / max(1, p["total_bytes"]) * 100
        if pct - last[0] >= 2 or p["phase"] in ("완료", "로컬 캐시에서 복사", "검증"):
            last[0] = pct
            print(f"[{pct:5.1f}%] {p['model']} {p['phase']} {p['file']}", flush=True)

    try:
        done = models.install(args.keys or None, progress=progress, allow_network=not args.offline)
    except models.ModelError as exc:
        print(f"설치 실패: {exc}", file=sys.stderr)
        return 1
    print("설치 완료:", done or "(이미 모두 설치됨)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
