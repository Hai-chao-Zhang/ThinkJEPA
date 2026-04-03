# ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model
# Copyright (c) 2024-2026 Northeastern University.
# Developed in NEU SMILE LAB by Haichao Zhang (https://zhanghaichao.xyz)
# and Yun Raymond Fu (https://www1.ece.neu.edu/~yunfu/).
# SPDX-style identifier: LicenseRef-ThinkJEPA-Attribution
# Original source: https://github.com/Hai-chao-Zhang/ThinkJEPA
# See the root LICENSE, NOTICE, CITATION.cff, and CITATION.bib for attribution and citation requirements.

from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULT_HF_HOME = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
DEFAULT_EGODEX_HF_REPO = os.environ.get("EGODEX_HF_REPO", "haichaozhang/cache")
_EGODEX_SPLITS = {"part1", "part2", "part3", "part4", "part5", "test", "extra"}
_LEGACY_EGODEX_ROOTS = tuple(
    p for p in os.environ.get("EGODEX_LEGACY_ROOTS", "").split(os.pathsep) if p
)
_TOKEN_ENV_VARS = (
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)


def get_huggingface_access_token() -> str | None:
    for name in _TOKEN_ENV_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def configure_huggingface_cache_dirs(cache_home: str | None = None) -> str:
    hf_home = os.path.abspath(
        os.path.expanduser(cache_home or os.environ.get("HF_HOME", DEFAULT_HF_HOME))
    )
    hub_cache = os.path.join(hf_home, "hub")
    datasets_cache = os.path.join(hf_home, "datasets")
    transformers_cache = os.path.join(hf_home, "transformers")

    os.environ["HF_HOME"] = hf_home
    os.environ.setdefault("HF_HUB_CACHE", hub_cache)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.environ["HF_HUB_CACHE"])
    os.environ.setdefault("HF_DATASETS_CACHE", datasets_cache)
    os.environ.setdefault("TRANSFORMERS_CACHE", transformers_cache)

    token = get_huggingface_access_token()
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGINGFACE_TOKEN", token)
    return hf_home


def _canonicalize_local_path(path: str) -> str:
    return os.path.realpath(os.path.abspath(os.path.expanduser(path)))


def default_egodex_hf_reference(subpath: str = "part2") -> str:
    repo_id = os.environ.get("EGODEX_HF_REPO", DEFAULT_EGODEX_HF_REPO).strip("/")
    spec = f"hf://datasets/{repo_id}"
    subpath = subpath.strip("/")
    if subpath:
        spec = f"{spec}/{subpath}"
    return spec


DEFAULT_EGODEX_HF_ROOT = default_egodex_hf_reference("")
DEFAULT_EGODEX_PART2_HF_DIR = default_egodex_hf_reference("part2")


def _is_huggingface_reference(path: str) -> bool:
    return isinstance(path, str) and path.startswith("hf://")


def _parse_huggingface_reference(path: str) -> tuple[str, str, str]:
    if not _is_huggingface_reference(path):
        raise ValueError(f"not a huggingface spec: {path}")
    parts = [part for part in path[len("hf://") :].split("/") if part]
    if len(parts) < 3:
        raise ValueError(
            "huggingface path must look like hf://datasets/<namespace>/<name>[/subdir]"
        )
    repo_kind = parts[0]
    repo_id = f"{parts[1]}/{parts[2]}"
    subpath = "/".join(parts[3:])
    if repo_kind == "datasets":
        repo_type = "dataset"
    elif repo_kind == "models":
        repo_type = "model"
    elif repo_kind == "spaces":
        repo_type = "space"
    else:
        raise ValueError(f"unsupported huggingface repo kind: {repo_kind}")
    return repo_type, repo_id, subpath


def resolve_huggingface_reference(path: str) -> str:
    repo_type, repo_id, subpath = _parse_huggingface_reference(path)
    configure_huggingface_cache_dirs()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to resolve hf:// paths"
        ) from exc

    allow_patterns = None
    if subpath:
        normalized_subpath = subpath.rstrip("/")
        allow_patterns = [
            normalized_subpath,
            f"{normalized_subpath}/*",
            f"{normalized_subpath}/**",
        ]

    snapshot_root = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        cache_dir=os.environ["HF_HUB_CACHE"],
        token=get_huggingface_access_token(),
        allow_patterns=allow_patterns,
    )
    if subpath:
        return os.path.normpath(os.path.join(snapshot_root, subpath))
    return os.path.normpath(snapshot_root)


def _is_legacy_egodex_reference(path: str) -> bool:
    norm = os.path.normpath(path)
    return any(
        norm == root or norm.startswith(root + os.sep) for root in _LEGACY_EGODEX_ROOTS
    )


def _candidate_egodex_base_roots(resolved_data_root: str) -> list[str]:
    roots = [os.path.normpath(resolved_data_root)]
    leaf = os.path.basename(os.path.normpath(resolved_data_root))
    if leaf in _EGODEX_SPLITS:
        roots.append(os.path.normpath(os.path.dirname(resolved_data_root)))
    out: list[str] = []
    for root in roots:
        if root not in out:
            out.append(root)
    return out


def _candidate_release_relative_variants(rel_path: str, current_leaf: str | None) -> list[str]:
    rel = os.path.normpath(rel_path)
    variants: list[str] = []

    def add(candidate: str):
        candidate = os.path.normpath(candidate)
        if candidate not in variants:
            variants.append(candidate)

    add(rel)
    parts = list(Path(rel).parts)
    if len(parts) >= 2 and parts[0] == parts[1]:
        add(os.path.join(*parts[1:]))
    if current_leaf and parts and parts[0] == current_leaf and len(parts) >= 2:
        add(os.path.join(*parts[1:]))
    if current_leaf and len(parts) >= 3 and parts[0] == current_leaf and parts[1] == current_leaf:
        add(os.path.join(*parts[2:]))
    return variants


def rewrite_legacy_egodex_reference(path: str, current_data_root: str) -> str:
    if not isinstance(path, str) or not path:
        return path
    if os.path.exists(path):
        return os.path.abspath(path)
    if not _is_legacy_egodex_reference(path):
        return path

    resolved_data_root = resolve_egodex_data_reference(current_data_root)
    current_leaf = os.path.basename(os.path.normpath(resolved_data_root))
    fallback: list[str] = []

    for legacy_root in _LEGACY_EGODEX_ROOTS:
        try:
            rel = os.path.relpath(path, legacy_root)
        except ValueError:
            continue
        if rel == ".." or rel.startswith(f"..{os.sep}"):
            continue
        for rel_variant in _candidate_release_relative_variants(rel, current_leaf):
            for base_root in _candidate_egodex_base_roots(resolved_data_root):
                candidate = os.path.normpath(os.path.join(base_root, rel_variant))
                if candidate not in fallback:
                    fallback.append(candidate)
                if os.path.exists(candidate):
                    return candidate
    return fallback[0] if fallback else path


def rewrite_manifest_paths_for_release(paths: list[str], current_data_root: str) -> list[str]:
    return [rewrite_legacy_egodex_reference(path, current_data_root) for path in paths]


def resolve_egodex_data_reference(path: str) -> str:
    if not isinstance(path, str):
        return path
    if _is_huggingface_reference(path):
        return resolve_huggingface_reference(path)
    if os.path.exists(path):
        return os.path.abspath(path)
    if _is_legacy_egodex_reference(path):
        root = resolve_huggingface_reference(DEFAULT_EGODEX_HF_ROOT)
        return rewrite_legacy_egodex_reference(path, root)
    return path


def is_huggingface_cache_path(path: str, cache_home: str | None = None) -> bool:
    if not isinstance(path, str) or not path:
        return False
    configure_huggingface_cache_dirs(cache_home)
    hf_home = _canonicalize_local_path(os.environ["HF_HOME"])
    path_real = _canonicalize_local_path(path)
    return path_real == hf_home or path_real.startswith(hf_home + os.sep)


def main() -> None:
    parser = argparse.ArgumentParser("Resolve EgoDex Hugging Face paths to local paths")
    parser.add_argument("path", nargs="?", default=DEFAULT_EGODEX_PART2_HF_DIR)
    args = parser.parse_args()
    configure_huggingface_cache_dirs()
    print(resolve_egodex_data_reference(args.path))


if __name__ == "__main__":
    main()
