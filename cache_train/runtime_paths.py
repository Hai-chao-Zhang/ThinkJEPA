"""Local path and model-cache configuration."""

from __future__ import annotations

import os
from pathlib import Path


def configure_huggingface_cache_dirs(cache_home: str) -> str:
    """Configure Hugging Face caches on an explicit non-home data volume."""
    value = str(cache_home or "").strip()
    if not value:
        raise ValueError("an explicit non-home Hugging Face cache root is required")
    hf_home = Path(value).expanduser().resolve(strict=False)
    user_home = Path.home().resolve()
    if hf_home == user_home or user_home in hf_home.parents:
        raise ValueError(f"Hugging Face cache root must be outside home: {hf_home}")

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.environ["HF_HUB_CACHE"]
    os.environ["HF_DATASETS_CACHE"] = str(hf_home / "datasets")
    os.environ["TRANSFORMERS_CACHE"] = str(hf_home / "transformers")
    return str(hf_home)


def resolve_egodex_data_reference(path: str) -> str:
    """Resolve an explicit local path and reject mutable remote references."""
    if not isinstance(path, str):
        return path
    value = path.strip()
    if value.startswith("hf://"):
        raise ValueError(
            "mutable hf:// roots are not supported; materialize an immutable "
            "dataset revision locally before training"
        )
    return str(Path(value).expanduser().resolve(strict=False))


def rewrite_manifest_paths(
    paths: list[str], current_data_root: str
) -> list[str]:
    """Resolve HDF5 manifest entries beneath the explicit local data root."""
    root = Path(resolve_egodex_data_reference(current_data_root))
    resolved = []
    for value in paths:
        candidate = Path(str(value)).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve(strict=False)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"manifest entry escapes the explicit data root: {value!r}"
            ) from exc
        resolved.append(str(candidate))
    return resolved
