from __future__ import annotations

import os
from pathlib import Path


DEFAULT_WORK_ROOT = Path(
    os.getenv("PHOMT_WORK_ROOT", r"D:\Users\Admin\Downloads\PhoMT")
).resolve()
DATA_DIR = Path(
    os.getenv("PHOMT_DATA_DIR", str(DEFAULT_WORK_ROOT / "detokenization"))
).resolve()
RESULTS_DIR = Path(
    os.getenv("PHOMT_RESULTS_DIR", str(DEFAULT_WORK_ROOT / "results"))
).resolve()
MODEL_DIR = Path(
    os.getenv("PHOMT_MODEL_DIR", str(DEFAULT_WORK_ROOT / "final_model_it"))
).resolve()
CACHE_ROOT = Path(
    os.getenv("PHOMT_CACHE_ROOT", str(DEFAULT_WORK_ROOT / ".cache"))
).resolve()
TMP_DIR = Path(
    os.getenv("PHOMT_TMP_DIR", str(DEFAULT_WORK_ROOT / ".tmp"))
).resolve()


def configure_runtime_dirs() -> None:
    hf_home = CACHE_ROOT / "huggingface"
    datasets_cache = hf_home / "datasets"
    hub_cache = hf_home / "hub"
    torch_cache = CACHE_ROOT / "torch"
    pip_cache = CACHE_ROOT / "pip"

    for path in (
        DEFAULT_WORK_ROOT,
        DATA_DIR,
        RESULTS_DIR,
        MODEL_DIR,
        CACHE_ROOT,
        TMP_DIR,
        hf_home,
        datasets_cache,
        hub_cache,
        torch_cache,
        pip_cache,
    ):
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(datasets_cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub_cache))
    os.environ.setdefault("TORCH_HOME", str(torch_cache))
    os.environ.setdefault("PIP_CACHE_DIR", str(pip_cache))
    os.environ.setdefault("TMP", str(TMP_DIR))
    os.environ.setdefault("TEMP", str(TMP_DIR))


def resolve_data_file(*parts: str) -> str:
    return str(DATA_DIR.joinpath(*parts))
