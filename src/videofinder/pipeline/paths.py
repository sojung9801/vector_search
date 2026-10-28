"""데이터 위치. 표시 이름(영상찾기)과 무관한 안정적 내부 식별자를 사용한다."""

from __future__ import annotations

import os
from pathlib import Path

APP_ID = "com.local.videofinder"  # bundle identifier 및 데이터 디렉터리 이름 (표시 이름 변경과 분리)
APP_DISPLAY_NAME = "영상찾기"
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def home() -> Path:
    override = os.environ.get("VF_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / "Library" / "Application Support" / APP_ID


def models_dir() -> Path:
    return home() / "models"


def library_dir() -> Path:
    return home() / "library"


def videos_dir() -> Path:
    return library_dir() / "videos"


def work_dir() -> Path:
    return home() / "work"


def state_dir() -> Path:
    return home() / "state"


def experiments_dir() -> Path:
    return home() / "experiments"


def logs_dir() -> Path:
    return home() / "logs"


def ensure_dirs() -> None:
    for d in (models_dir(), videos_dir(), work_dir(), state_dir(), experiments_dir(), logs_dir()):
        d.mkdir(parents=True, exist_ok=True)


def config_dir() -> Path:
    return PROJECT_ROOT / "configs"


def default_config_path() -> Path:
    return config_dir() / "default.yaml"
