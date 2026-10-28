"""모델 레지스트리와 설치기.

- configs/models.lock.json 에 고정된 revision·파일·크기·해시만 사용한다 (런타임에 최신 버전 조회 없음).
- 설치: (1) 기존 Hugging Face 캐시에 동일 해시 파일이 있으면 복제, (2) 없으면 Range 이어받기 다운로드.
- 모든 파일을 sha256(LFS) 또는 git blob sha1(작은 파일)로 검증한 뒤 staging → 최종 경로로 원자적 이동.
- 설치 완료 표시는 `.complete.json`. 이것이 없는 디렉터리는 완성 모델로 취급하지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from videofinder.pipeline import paths
from videofinder.pipeline.jsonio import read_json, write_json

LOCK_PATH = paths.config_dir() / "models.lock.json"
COMPLETE_MARKER = ".complete.json"
HF_BASE = "https://huggingface.co"


class ModelError(RuntimeError):
    """모델 미설치·검증 실패·다운로드 실패 (한국어 메시지)."""


@dataclass(frozen=True)
class ModelFile:
    path: str
    size: int
    sha256: str | None
    git_sha1: str | None
    url: str | None = None


@dataclass(frozen=True)
class ModelSpec:
    key: str
    role: str
    source: str
    repo: str | None
    revision: str
    files: tuple[ModelFile, ...]
    license: str
    license_url: str
    grade: str
    required: bool

    @property
    def size(self) -> int:
        return sum(f.size for f in self.files)

    @property
    def name(self) -> str:
        return self.repo or self.files[0].path

    def file_url(self, f: ModelFile) -> str:
        if f.url:
            return f.url
        return f"{HF_BASE}/{self.repo}/resolve/{self.revision}/{f.path}"


def load_specs(lock_path: Path = LOCK_PATH) -> dict[str, ModelSpec]:
    data = read_json(lock_path)
    specs: dict[str, ModelSpec] = {}
    for e in data["models"]:
        specs[e["key"]] = ModelSpec(
            key=e["key"],
            role=e["role"],
            source=e["source"],
            repo=e.get("repo"),
            revision=e["revision"],
            files=tuple(ModelFile(path=f["path"], size=int(f["size"]), sha256=f.get("sha256"), git_sha1=f.get("git_sha1"), url=f.get("url")) for f in e["files"]),
            license=e["license"],
            license_url=e["license_url"],
            grade=e["grade"],
            required=bool(e.get("required", True)),
        )
    return specs


def model_dir(spec: ModelSpec) -> Path:
    return paths.models_dir() / spec.key / spec.revision[:12]


def is_installed(spec: ModelSpec) -> bool:
    marker = model_dir(spec) / COMPLETE_MARKER
    if not marker.is_file():
        return False
    try:
        info = read_json(marker)
    except (OSError, json.JSONDecodeError):
        return False
    if info.get("revision") != spec.revision:
        return False
    for f in spec.files:
        p = model_dir(spec) / f.path
        if not p.is_file() or p.stat().st_size != f.size:
            return False
    return True


def model_path(key: str, specs: dict[str, ModelSpec] | None = None) -> Path:
    """설치된 모델 디렉터리. 미설치면 ModelError (자동 다운로드 금지 — 설정 화면에서만 설치)."""
    spec = (specs or load_specs())[key]
    if not is_installed(spec):
        raise ModelError(f"모델이 설치되지 않았습니다: {spec.name} ({spec.role}). 설정 화면에서 모델을 설치하세요.")
    return model_dir(spec)


def status(specs: dict[str, ModelSpec] | None = None) -> list[dict[str, Any]]:
    specs = specs or load_specs()
    out = []
    for spec in specs.values():
        out.append(
            {
                "key": spec.key,
                "name": spec.name,
                "role": spec.role,
                "revision": spec.revision,
                "size": spec.size,
                "license": spec.license,
                "license_url": spec.license_url,
                "grade": spec.grade,
                "required": spec.required,
                "installed": is_installed(spec),
                "path": str(model_dir(spec)),
                "partial_bytes": _partial_bytes(spec),
                "cached_elsewhere": _hf_cache_hits(spec),
            }
        )
    return out


def all_required_installed(specs: dict[str, ModelSpec] | None = None) -> bool:
    specs = specs or load_specs()
    return all(is_installed(s) for s in specs.values() if s.required)


# ---------- 검증 ----------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def _git_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def verify_file(path: Path, f: ModelFile) -> bool:
    if not path.is_file() or path.stat().st_size != f.size:
        return False
    if f.sha256:
        return _sha256(path) == f.sha256
    if f.git_sha1:
        return _git_sha1(path) == f.git_sha1
    return True


# ---------- HF 캐시 재사용 ----------

def _hf_cache_root() -> Path:
    hf_home = os.environ.get("HF_HOME")
    return Path(hf_home) / "hub" if hf_home else Path.home() / ".cache" / "huggingface" / "hub"


def _hf_cache_candidate(spec: ModelSpec, f: ModelFile) -> Path | None:
    if spec.source != "huggingface" or not spec.repo:
        return None
    repo_dir = _hf_cache_root() / ("models--" + spec.repo.replace("/", "--"))
    if f.sha256:
        blob = repo_dir / "blobs" / f.sha256
        return blob if blob.is_file() and blob.stat().st_size == f.size else None
    snaps = repo_dir / "snapshots"
    if snaps.is_dir():
        for snap in snaps.iterdir():
            p = snap / f.path
            if p.is_file() and p.stat().st_size == f.size:
                return p
    return None


def _hf_cache_hits(spec: ModelSpec) -> int:
    return sum(f.size for f in spec.files if _hf_cache_candidate(spec, f) is not None)


def _partial_dir(spec: ModelSpec) -> Path:
    return paths.models_dir() / ".partial" / spec.key / spec.revision[:12]


def _partial_bytes(spec: ModelSpec) -> int:
    d = _partial_dir(spec)
    return sum(p.stat().st_size for p in d.rglob("*") if p.is_file()) if d.is_dir() else 0


def _clone(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".clone")
    proc = subprocess.run(["/bin/cp", "-c", str(src.resolve()), str(tmp)], capture_output=True, check=False)
    if proc.returncode != 0:
        shutil.copyfile(src.resolve(), tmp)
    os.replace(tmp, dst)


# ---------- 다운로드 ----------

ProgressFn = Callable[[dict[str, Any]], None]


class InstallCancelled(RuntimeError):
    pass


def _download(url: str, part: Path, expected_size: int, on_bytes: Callable[[int], None], cancel: Callable[[], bool]) -> None:
    import httpx

    part.parent.mkdir(parents=True, exist_ok=True)
    have = part.stat().st_size if part.is_file() else 0
    if have > expected_size:
        part.unlink()
        have = 0
    if have == expected_size:
        return
    headers = {"Range": f"bytes={have}-"} if have else {}
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0, connect=20.0)) as client:
        with client.stream("GET", url, headers=headers) as resp:
            if resp.status_code == 200 and have:
                have = 0  # 서버가 Range 미지원 → 처음부터
                part.unlink(missing_ok=True)
            elif resp.status_code not in (200, 206):
                raise ModelError(f"다운로드 실패 HTTP {resp.status_code}: {url}")
            mode = "ab" if have else "wb"
            with part.open(mode) as fh:
                for chunk in resp.iter_bytes(1024 * 1024):
                    if cancel():
                        raise InstallCancelled("설치 취소됨")
                    fh.write(chunk)
                    on_bytes(len(chunk))


def install(keys: list[str] | None = None, progress: ProgressFn | None = None, cancel: Callable[[], bool] | None = None, allow_network: bool = True) -> list[str]:
    """선택 모델 설치. 반환: 새로 설치한 key 목록."""
    specs = load_specs()
    targets = [specs[k] for k in (keys or list(specs))]
    cancel = cancel or (lambda: False)
    total = sum(s.size for s in targets if not is_installed(s))
    done = 0
    started = time.monotonic()
    installed: list[str] = []

    def report(spec: ModelSpec, file: str, phase: str) -> None:
        if progress:
            progress({"model": spec.key, "name": spec.name, "file": file, "phase": phase, "done_bytes": done, "total_bytes": total, "elapsed": round(time.monotonic() - started, 1)})

    for spec in targets:
        if is_installed(spec):
            continue
        final = model_dir(spec)
        staging = paths.models_dir() / ".staging" / f"{spec.key}-{spec.revision[:12]}"
        staging.mkdir(parents=True, exist_ok=True)
        for f in spec.files:
            dst = staging / f.path
            if dst.is_file() and verify_file(dst, f):
                done += f.size
                report(spec, f.path, "재사용")
                continue
            cached = _hf_cache_candidate(spec, f)
            if cached is not None:
                report(spec, f.path, "로컬 캐시에서 복사")
                _clone(cached, dst)
                if verify_file(dst, f):
                    done += f.size
                    continue
                dst.unlink(missing_ok=True)
            if not allow_network:
                raise ModelError(f"네트워크 없이 설치할 수 없는 파일: {spec.name}/{f.path}")
            part = _partial_dir(spec) / (f.path + ".part")
            before = part.stat().st_size if part.is_file() else 0
            done += min(before, f.size)

            def on_bytes(n: int, _spec: ModelSpec = spec, _f: ModelFile = f) -> None:
                nonlocal done
                done += n
                report(_spec, _f.path, "다운로드")

            _download(spec.file_url(f), part, f.size, on_bytes, cancel)
            report(spec, f.path, "검증")
            if not verify_file(part, f):
                part.unlink(missing_ok=True)
                raise ModelError(f"무결성 검증 실패 (다시 시도하면 새로 받습니다): {spec.name}/{f.path}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(part, dst)
        # 전체 검증 후 원자적 이동
        for f in spec.files:
            if not verify_file(staging / f.path, f):
                raise ModelError(f"설치 검증 실패: {spec.name}/{f.path}")
        write_json(staging / COMPLETE_MARKER, {"key": spec.key, "name": spec.name, "revision": spec.revision, "installed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "files": [f.path for f in spec.files]})
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            shutil.rmtree(final)
        os.rename(staging, final)
        shutil.rmtree(_partial_dir(spec), ignore_errors=True)
        installed.append(spec.key)
        report(spec, "", "완료")
    return installed


def disk_usage() -> int:
    root = paths.models_dir()
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.is_dir() else 0
