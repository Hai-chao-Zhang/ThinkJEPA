#!/usr/bin/env python3
"""Validate an immutable ThinkJEPA causal feature cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--expected_count", type=int, default=2000)
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def relative_map(root: Path, suffix: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in sorted(root.rglob(f"*{suffix}")):
        if path.name.startswith("."):
            continue
        rel = path.relative_to(root).with_suffix("").as_posix()
        if rel in out:
            raise ValueError(f"duplicate relative stem {rel!r}: {out[rel]} and {path}")
        out[rel] = path
    return out


def scalar(payload: dict[str, np.ndarray], key: str) -> object:
    value = np.asarray(payload[key])
    if value.size != 1:
        raise ValueError(f"{key} must be scalar, got {value.shape}")
    item = value.reshape(-1)[0]
    return item.item() if hasattr(item, "item") else item


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    # Keep the CLI help/import path lightweight; the dataset module also imports
    # the video backend, which is needed only for an actual validation run.
    from egodex.trajectory_dataset import validate_causal_cache_payload

    raw_root = Path(args.raw_dir).expanduser().resolve()
    cache_root = Path(args.cache_dir).expanduser().resolve()
    report_path = Path(args.output_json).expanduser().resolve()
    raw = relative_map(raw_root, ".mp4")
    cache = relative_map(cache_root, ".npz")

    expected_count = int(args.expected_count)
    if len(raw) != expected_count or len(cache) != expected_count:
        raise ValueError(
            f"expected {expected_count} raw/cache samples, got raw={len(raw)} cache={len(cache)}"
        )
    if set(raw) != set(cache):
        missing_cache = sorted(set(raw) - set(cache))[:20]
        missing_raw = sorted(set(cache) - set(raw))[:20]
        raise ValueError(
            f"raw/cache relative-stem mismatch: missing_cache={missing_cache} "
            f"missing_raw={missing_raw}"
        )

    fingerprints: set[str] = set()
    qwen_checkpoints: set[str] = set()
    vjepa_checkpoints: set[str] = set()
    old_lengths: Counter[int] = Counter()
    new_lengths: Counter[int] = Counter()
    token_lengths: Counter[int] = Counter()
    raw_content_hashes: set[str] = set()
    raw_manifest_digest = hashlib.sha256()
    cache_file_set_digest = hashlib.sha256()
    total_cache_bytes = 0
    started = time.time()

    for index, rel_stem in enumerate(sorted(raw), start=1):
        raw_path = raw[rel_stem]
        cache_path = cache[rel_stem]
        with np.load(cache_path, allow_pickle=False) as archive:
            payload = {key: archive[key] for key in archive.files}
        validate_causal_cache_payload(payload, path=str(cache_path))

        expected_vjepa_shape = (16, 256, 1024)
        for key in ("vjepa_input_feats", "vjepa_target_feats"):
            if (
                tuple(payload[key].shape) != expected_vjepa_shape
                or payload[key].dtype != np.float16
            ):
                raise ValueError(
                    f"{key} at {cache_path} must be float16 {expected_vjepa_shape}, "
                    f"got {payload[key].dtype} {payload[key].shape}"
                )
        imgs = np.asarray(payload.get("imgs"))
        if imgs.shape != (64, 256, 256, 3) or imgs.dtype != np.uint8:
            raise ValueError(
                f"invalid cached raw observation tensor at {cache_path}: "
                f"shape={imgs.shape} dtype={imgs.dtype}"
            )
        expected_layers = np.asarray([0, 4, 8, 12, 16, 20, 24, 27], dtype=np.int32)
        layers = np.asarray(payload.get("layers", []), dtype=np.int32).reshape(-1)
        if not np.array_equal(layers, expected_layers):
            raise ValueError(
                f"cache {cache_path} does not contain the complete paper VLM layer set: "
                f"{layers.tolist()}"
            )
        if (
            payload["vlm_old"].shape[0] != expected_layers.size
            or payload["vlm_new"].shape[0] != expected_layers.size
            or payload["vlm_old"].shape[-1] != 2048
            or payload["vlm_new"].shape[-1] != 2048
            or payload["vlm_old"].dtype != np.float16
            or payload["vlm_new"].dtype != np.float16
        ):
            raise ValueError(
                f"invalid VLM feature width/layer layout at {cache_path}: "
                f"old={payload['vlm_old'].shape} new={payload['vlm_new'].shape}"
            )

        expected_relpath = f"{rel_stem}.mp4"
        if str(scalar(payload, "source_video_relpath")) != expected_relpath:
            raise ValueError(
                f"source relpath mismatch for {cache_path}: "
                f"{scalar(payload, 'source_video_relpath')!r} != {expected_relpath!r}"
            )
        stat = raw_path.stat()
        raw_content_sha256 = sha256_file(raw_path)
        if raw_content_sha256 in raw_content_hashes:
            raise ValueError(
                f"duplicate raw-video content detected at {raw_path}; refusing a "
                "potential cross-split duplicate"
            )
        raw_content_hashes.add(raw_content_sha256)
        raw_manifest_digest.update(
            f"{expected_relpath}\t{raw_content_sha256}\n".encode("utf-8")
        )
        if int(scalar(payload, "source_size_bytes")) != int(stat.st_size):
            raise ValueError(f"source size changed for {raw_path}")
        if int(scalar(payload, "source_mtime_ns")) != int(stat.st_mtime_ns):
            raise ValueError(f"source mtime changed for {raw_path}")
        for key in ("vjepa_input_feats", "vjepa_target_feats", "vlm_old", "vlm_new"):
            if not np.all(np.isfinite(payload[key])):
                raise ValueError(f"non-finite {key} in {cache_path}")

        fingerprints.add(str(scalar(payload, "cache_config_fingerprint")))
        qwen_checkpoints.add(str(scalar(payload, "qwen_checkpoint_sha")))
        vjepa_checkpoints.add(str(scalar(payload, "vjepa_checkpoint_sha256")))
        old_lengths[int(payload["vlm_old"].shape[1])] += 1
        new_lengths[int(payload["vlm_new"].shape[1])] += 1
        token_lengths[int(payload["token_ids"].size)] += 1
        cache_size = int(cache_path.stat().st_size)
        total_cache_bytes += cache_size
        cache_file_set_digest.update(
            f"{rel_stem}.npz\t{cache_size}\n".encode("utf-8")
        )
        if index % 100 == 0 or index == expected_count:
            print(f"[VALIDATE] {index}/{expected_count}", flush=True)

    if len(fingerprints) != 1:
        raise ValueError(f"mixed configuration fingerprints: {sorted(fingerprints)}")
    if len(qwen_checkpoints) != 1 or len(vjepa_checkpoints) != 1:
        raise ValueError(
            f"mixed model checkpoints: qwen={sorted(qwen_checkpoints)} "
            f"vjepa={sorted(vjepa_checkpoints)}"
        )

    temporary_files = sorted(
        str(path.relative_to(cache_root))
        for path in cache_root.rglob(".*.tmp.*")
        if path.is_file()
    )
    if temporary_files:
        raise ValueError(f"temporary cache files remain: {temporary_files[:20]}")

    # The validated cache is an immutable feature store.  Removing write bits
    # prevents accidental in-place edits while keeping deletion/rebuild under
    # the writable parent directory possible.
    for cache_path in cache.values():
        cache_path.chmod(cache_path.stat().st_mode & ~0o222)
    writable_after_validation = [
        str(path.relative_to(cache_root))
        for path in cache.values()
        if path.stat().st_mode & 0o222
    ]
    if writable_after_validation:
        raise ValueError(
            f"validated cache files remain writable: {writable_after_validation[:20]}"
        )

    report: dict[str, object] = {
        "status": "ok",
        "schema": "thinkjepa.causal_split.v2",
        "extractor_version": "2.0.5",
        "raw_count": len(raw),
        "cache_count": len(cache),
        "configuration_fingerprint": next(iter(fingerprints)),
        "qwen_checkpoint_sha": next(iter(qwen_checkpoints)),
        "vjepa_checkpoint_sha256": next(iter(vjepa_checkpoints)),
        "vlm_old_length_histogram": dict(sorted(old_lengths.items())),
        "vlm_new_length_histogram": dict(sorted(new_lengths.items())),
        "token_id_length_histogram": dict(sorted(token_lengths.items())),
        "total_cache_bytes": total_cache_bytes,
        "cache_file_set_sha256": cache_file_set_digest.hexdigest(),
        "raw_content_duplicate_count": 0,
        "raw_content_manifest_sha256": raw_manifest_digest.hexdigest(),
        "cache_files_write_protected": True,
        "elapsed_seconds": time.time() - started,
        "validated_unix": time.time(),
        "raw_root": str(raw_root),
        "cache_root": str(cache_root),
    }
    write_json_atomic(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
