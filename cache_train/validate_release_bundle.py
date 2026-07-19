#!/usr/bin/env python3
"""Validate a local portable ThinkJEPA cache and supervision bundle.

This tool never copies data and has no Hugging Face or upload integration.  It
rehashes the portable HDF5 supervision, validates every causal feature archive,
checks the exact train/test/cache file set, and optionally writes the two
bundle-relative contracts consumed by ``cache_train/thinker_train.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cache_train.portable_split import (  # noqa: E402
    PORTABLE_SPLIT_SCHEMA,
    normalize_relative_posix_path,
    sha256_file,
    validate_portable_split_bundle,
)


CACHE_SCHEMA = "thinkjepa.causal_split.v2"
CACHE_EXTRACTOR_VERSION = "2.0.5"
RELEASE_MARKER_SCHEMA = "thinkjepa.cache_validation.portable.v1"
HEX_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate an already materialized local bundle containing cache/, "
            "supervision_hdf5/, and manifests/portable_v1/."
        )
    )
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument(
        "--source-cache-report",
        help=(
            "source cache validation report; required with --write-contract because "
            "a feature-only bundle cannot independently recheck raw-video duplicates"
        ),
    )
    parser.add_argument("--expected-count", type=int, default=2000)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--write-contract",
        action="store_true",
        help="atomically write bundle-root full_validation.json and VALIDATED_SUCCESS",
    )
    mode.add_argument(
        "--verify-only",
        action="store_true",
        help="verify existing contracts without writing any file",
    )
    return parser.parse_args()


def load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"cannot read {label} JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    text = json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def scalar(payload: Mapping[str, Any], key: str) -> object:
    if key not in payload:
        raise KeyError(f"causal feature is missing {key!r}")
    value = np.asarray(payload[key])
    if value.size != 1:
        raise ValueError(f"{key} must be scalar, got shape={value.shape}")
    item = value.reshape(-1)[0]
    if hasattr(item, "item"):
        item = item.item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    return item


def validate_sha256(value: object, *, label: str) -> str:
    digest = str(value)
    if HEX_SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"invalid SHA-256 for {label}")
    return digest


def assert_no_absolute_paths(value: object, *, label: str = "contract") -> None:
    """Reject host paths in recursively generated contracts."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_no_absolute_paths(item, label=f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_no_absolute_paths(item, label=f"{label}[{index}]")
        return
    if not isinstance(value, str) or not value:
        return
    if value.startswith(("/", "\\\\")) or re.match(r"^[A-Za-z]:[\\/]", value):
        raise ValueError(f"absolute path is forbidden in {label}: {value!r}")
    if value.startswith("file://"):
        raise ValueError(f"file URI is forbidden in {label}: {value!r}")


def collect_relative_files(root: Path, *, suffix: str) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() != suffix.lower():
            continue
        relative = normalize_relative_posix_path(
            path.relative_to(root).as_posix(), expected_suffix=suffix
        )
        if relative in files:
            raise ValueError(f"duplicate portable path below {root}: {relative}")
        files[relative] = path
    return files


def validate_group_split(
    meta: Mapping[str, Any], train_relpaths: list[str], test_relpaths: list[str]
) -> None:
    if (
        meta.get("split_mode") != "group_aware"
        or bool(meta.get("sample_level_split", True))
        or int(meta.get("group_intersection_count", -1)) != 0
        or meta.get("group_intersection_assertion_passed") is not True
    ):
        raise ValueError("portable metadata does not prove a disjoint group-aware split")

    from cache_train.build_video_cache_splits import (
        hash_group_keys,
        infer_video_group_key,
    )

    dataset = str(meta.get("dataset", ""))
    group_regex = meta.get("group_regex") or None
    resolved_group_by = str(meta.get("group_by_resolved", ""))
    group_by = (
        str(meta.get("group_by_requested", "auto"))
        if resolved_group_by == "regex"
        else resolved_group_by
    )
    sentinel_root = "/portable-supervision-root"

    def group_keys(relative_paths: list[str]) -> set[str]:
        return {
            infer_video_group_key(
                dataset,
                str(Path(sentinel_root) / Path(path).with_suffix(".mp4")),
                sentinel_root,
                group_by=group_by,
                group_regex=group_regex,
            )
            for path in relative_paths
        }

    train_groups = group_keys(train_relpaths)
    test_groups = group_keys(test_relpaths)
    intersection = train_groups & test_groups
    if (
        intersection
        or hash_group_keys(train_groups) != str(meta.get("train_group_hash", ""))
        or hash_group_keys(test_groups) != str(meta.get("test_group_hash", ""))
    ):
        raise ValueError(
            "portable manifest-derived groups differ from split metadata: "
            f"intersection={sorted(intersection)[:10]}"
        )


def validate_feature_cache(
    cache_root: Path,
    *,
    expected_relpaths: set[str],
    expected_count: int,
    expected_fingerprint: str,
) -> dict[str, Any]:
    # Keep --help lightweight.  The dataset module imports the video backend,
    # although only its schema-v2 validator is used here.
    from egodex.trajectory_dataset import validate_causal_cache_payload

    cache_files = collect_relative_files(cache_root, suffix=".npz")
    actual_relpaths = set(cache_files)
    if actual_relpaths != expected_relpaths:
        missing = sorted(expected_relpaths - actual_relpaths)[:20]
        extra = sorted(actual_relpaths - expected_relpaths)[:20]
        raise ValueError(
            "cache file set differs from portable manifests: "
            f"missing={missing} extra={extra}"
        )
    if len(cache_files) != expected_count:
        raise ValueError(
            f"expected {expected_count} causal archives, found {len(cache_files)}"
        )

    fingerprints: set[str] = set()
    qwen_checkpoints: set[str] = set()
    vjepa_checkpoints: set[str] = set()
    old_lengths: Counter[int] = Counter()
    new_lengths: Counter[int] = Counter()
    token_lengths: Counter[int] = Counter()
    total_cache_bytes = 0
    file_set_digest = hashlib.sha256()

    for index, (relative, path) in enumerate(sorted(cache_files.items()), start=1):
        if path.is_symlink():
            raise ValueError(f"causal feature must not be a symlink: {relative}")
        if path.stat().st_mode & 0o222:
            raise ValueError(f"causal feature is writable: {relative}")
        with np.load(path, allow_pickle=False) as archive:
            payload = {key: archive[key] for key in archive.files}
        validate_causal_cache_payload(payload, path=str(path))
        for key in ("vjepa_input_feats", "vjepa_target_feats", "vlm_old", "vlm_new"):
            values = np.asarray(payload[key])
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite {key} in {relative}")

        fingerprint = str(scalar(payload, "cache_config_fingerprint"))
        fingerprints.add(fingerprint)
        qwen_checkpoints.add(str(scalar(payload, "qwen_checkpoint_sha")))
        vjepa_checkpoints.add(str(scalar(payload, "vjepa_checkpoint_sha256")))
        old_lengths[int(np.asarray(payload["vlm_old"]).shape[1])] += 1
        new_lengths[int(np.asarray(payload["vlm_new"]).shape[1])] += 1
        token_lengths[int(np.asarray(payload["token_ids"]).size)] += 1
        size = int(path.stat().st_size)
        total_cache_bytes += size
        file_set_digest.update(f"{relative}\t{size}\n".encode("utf-8"))
        if index % 100 == 0 or index == expected_count:
            print(f"[BUNDLE-CACHE] {index}/{expected_count}", flush=True)

    if fingerprints != {expected_fingerprint}:
        raise ValueError(
            "causal cache configuration differs from portable metadata: "
            f"{sorted(fingerprints)}"
        )
    if len(qwen_checkpoints) != 1 or len(vjepa_checkpoints) != 1:
        raise ValueError(
            "mixed model checkpoint identities in causal cache: "
            f"qwen={sorted(qwen_checkpoints)} vjepa={sorted(vjepa_checkpoints)}"
        )
    return {
        "configuration_fingerprint": next(iter(fingerprints)),
        "qwen_checkpoint_sha": next(iter(qwen_checkpoints)),
        "vjepa_checkpoint_sha256": next(iter(vjepa_checkpoints)),
        "vlm_old_length_histogram": {
            str(key): value for key, value in sorted(old_lengths.items())
        },
        "vlm_new_length_histogram": {
            str(key): value for key, value in sorted(new_lengths.items())
        },
        "token_id_length_histogram": {
            str(key): value for key, value in sorted(token_lengths.items())
        },
        "total_cache_bytes": total_cache_bytes,
        "cache_file_set_sha256": file_set_digest.hexdigest(),
    }


def validate_source_cache_report(
    path: Path,
    *,
    expected_count: int,
    feature_summary: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    report = load_json_object(path, label="source cache report")
    if (
        report.get("status") != "ok"
        or report.get("schema") != CACHE_SCHEMA
        or report.get("extractor_version") != CACHE_EXTRACTOR_VERSION
        or int(report.get("raw_count", -1)) != expected_count
        or int(report.get("cache_count", -1)) != expected_count
        or int(report.get("raw_content_duplicate_count", -1)) != 0
        or report.get("cache_files_write_protected") is not True
    ):
        raise ValueError(f"source cache report is unsafe or incomplete: {path}")
    if report.get("configuration_fingerprint") != feature_summary.get(
        "configuration_fingerprint"
    ):
        raise ValueError("source cache report configuration differs from the bundle")
    for key in ("qwen_checkpoint_sha", "vjepa_checkpoint_sha256"):
        if str(report.get(key, "")) != str(feature_summary.get(key, "")):
            raise ValueError(f"source cache report {key} differs from the bundle")
    for key in (
        "total_cache_bytes",
        "vlm_old_length_histogram",
        "vlm_new_length_histogram",
        "token_id_length_histogram",
        "cache_file_set_sha256",
    ):
        if report.get(key) != feature_summary.get(key):
            raise ValueError(f"source cache report {key} differs from the bundle")
    validate_sha256(
        report.get("raw_content_manifest_sha256"),
        label="source raw-content manifest",
    )
    return report, sha256_file(path)


def validate_layout(bundle_root: Path, expected_count: int) -> dict[str, Any]:
    cache_root = bundle_root / "cache"
    supervision_root = bundle_root / "supervision_hdf5"
    manifest_root = bundle_root / "manifests" / "portable_v1"
    meta_path = manifest_root / "meta.json"
    train_manifest = manifest_root / "train_cache.txt"
    test_manifest = manifest_root / "test_cache.txt"
    for path in (cache_root, supervision_root, manifest_root):
        if not path.is_dir():
            raise FileNotFoundError(path)
    for path in (meta_path, train_manifest, test_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)

    meta = load_json_object(meta_path, label="portable split metadata")
    if meta.get("split_schema") != PORTABLE_SPLIT_SCHEMA:
        raise ValueError("bundle does not use the portable-v1 split schema")
    if int(meta.get("supervision_file_count", -1)) != expected_count:
        raise ValueError(
            "portable metadata count differs from --expected-count: "
            f"{meta.get('supervision_file_count')} != {expected_count}"
        )
    fingerprint = validate_sha256(
        meta.get("feature_config_fingerprint"),
        label="portable feature configuration",
    )
    portable = validate_portable_split_bundle(
        meta=meta,
        split_meta_path=meta_path,
        train_manifest=train_manifest,
        test_manifest=test_manifest,
        supervision_root=supervision_root,
        cache_root=cache_root,
        verify_supervision_content=True,
    )
    train_relpaths = list(portable["train_relpaths"])
    test_relpaths = list(portable["test_relpaths"])
    if len(train_relpaths) + len(test_relpaths) != expected_count:
        raise ValueError("portable train/test set does not match --expected-count")
    validate_group_split(meta, train_relpaths, test_relpaths)

    feature_summary = validate_feature_cache(
        cache_root,
        expected_relpaths=set(train_relpaths) | set(test_relpaths),
        expected_count=expected_count,
        expected_fingerprint=fingerprint,
    )
    return {
        "cache_root": cache_root,
        "supervision_root": supervision_root,
        "meta": meta,
        "meta_path": meta_path,
        "portable": portable,
        "feature_summary": feature_summary,
    }


def build_release_report(
    *,
    expected_count: int,
    validated: Mapping[str, Any],
    source_report: Mapping[str, Any],
    source_report_sha256: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    meta = validated["meta"]
    meta_path = validated["meta_path"]
    portable = validated["portable"]
    feature_summary = validated["feature_summary"]
    report: dict[str, Any] = {
        "status": "ok",
        "schema": CACHE_SCHEMA,
        "extractor_version": CACHE_EXTRACTOR_VERSION,
        "path_mode": "bundle_relative",
        "cache_root": "cache",
        "raw_count": expected_count,
        "cache_count": expected_count,
        "raw_content_duplicate_count": 0,
        "raw_content_manifest_sha256": validate_sha256(
            source_report.get("raw_content_manifest_sha256"),
            label="source raw-content manifest",
        ),
        "cache_files_write_protected": True,
        "configuration_fingerprint": feature_summary["configuration_fingerprint"],
        "qwen_checkpoint_sha": feature_summary["qwen_checkpoint_sha"],
        "vjepa_checkpoint_sha256": feature_summary["vjepa_checkpoint_sha256"],
        "vlm_old_length_histogram": feature_summary["vlm_old_length_histogram"],
        "vlm_new_length_histogram": feature_summary["vlm_new_length_histogram"],
        "token_id_length_histogram": feature_summary["token_id_length_histogram"],
        "total_cache_bytes": feature_summary["total_cache_bytes"],
        "cache_file_set_sha256": feature_summary["cache_file_set_sha256"],
        "portable_split_schema": PORTABLE_SPLIT_SCHEMA,
        "portable_split_meta_sha256": sha256_file(meta_path),
        "supervision_pairs_manifest_sha256": str(
            meta["supervision_pairs_manifest_sha256"]
        ),
        "supervision_content_set_sha256": str(
            meta["supervision_content_set_sha256"]
        ),
        "supervision_file_count": int(portable["supervision_file_count"]),
        "supervision_content_rehashed": portable["content_rehashed"] is True,
        "train_count": len(portable["train_relpaths"]),
        "test_count": len(portable["test_relpaths"]),
        "source_cache_report_sha256": source_report_sha256,
        "elapsed_seconds": elapsed_seconds,
        "validated_unix": time.time(),
    }
    assert_no_absolute_paths(report)
    return report


def build_success_marker(report_path: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    marker = {
        "schema": RELEASE_MARKER_SCHEMA,
        "full_validation_sha256": sha256_file(report_path),
        "portable_split_meta_sha256": report["portable_split_meta_sha256"],
        "supervision_pairs_manifest_sha256": report[
            "supervision_pairs_manifest_sha256"
        ],
        "source_cache_report_sha256": report["source_cache_report_sha256"],
        "configuration_fingerprint": report["configuration_fingerprint"],
        "cache_count": report["cache_count"],
    }
    assert_no_absolute_paths(marker)
    return marker


def verify_existing_contract(
    *,
    report_path: Path,
    marker_path: Path,
    expected_count: int,
    validated: Mapping[str, Any],
    source_report: Mapping[str, Any] | None,
    source_report_sha256: str | None,
) -> None:
    if not report_path.is_file() or not marker_path.is_file():
        raise FileNotFoundError(
            f"missing validation contract: report={report_path} marker={marker_path}"
        )
    report = load_json_object(report_path, label="bundle validation report")
    marker = load_json_object(marker_path, label="bundle success marker")
    assert_no_absolute_paths(report)
    assert_no_absolute_paths(marker)

    meta = validated["meta"]
    feature_summary = validated["feature_summary"]
    portable = validated["portable"]
    expected = {
        "status": "ok",
        "schema": CACHE_SCHEMA,
        "extractor_version": CACHE_EXTRACTOR_VERSION,
        "path_mode": "bundle_relative",
        "cache_root": "cache",
        "raw_count": expected_count,
        "cache_count": expected_count,
        "raw_content_duplicate_count": 0,
        "cache_files_write_protected": True,
        "configuration_fingerprint": feature_summary["configuration_fingerprint"],
        "qwen_checkpoint_sha": feature_summary["qwen_checkpoint_sha"],
        "vjepa_checkpoint_sha256": feature_summary["vjepa_checkpoint_sha256"],
        "vlm_old_length_histogram": feature_summary["vlm_old_length_histogram"],
        "vlm_new_length_histogram": feature_summary["vlm_new_length_histogram"],
        "token_id_length_histogram": feature_summary["token_id_length_histogram"],
        "total_cache_bytes": feature_summary["total_cache_bytes"],
        "cache_file_set_sha256": feature_summary["cache_file_set_sha256"],
        "portable_split_schema": PORTABLE_SPLIT_SCHEMA,
        "portable_split_meta_sha256": sha256_file(validated["meta_path"]),
        "supervision_pairs_manifest_sha256": str(
            meta["supervision_pairs_manifest_sha256"]
        ),
        "supervision_content_set_sha256": str(
            meta["supervision_content_set_sha256"]
        ),
        "supervision_file_count": int(portable["supervision_file_count"]),
        "supervision_content_rehashed": True,
        "train_count": len(portable["train_relpaths"]),
        "test_count": len(portable["test_relpaths"]),
    }
    mismatches = [
        f"{key}: report={report.get(key)!r} current={value!r}"
        for key, value in expected.items()
        if report.get(key) != value
    ]
    validate_sha256(
        report.get("raw_content_manifest_sha256"),
        label="bundle raw-content manifest",
    )
    recorded_source_sha256 = validate_sha256(
        report.get("source_cache_report_sha256"),
        label="bundle source cache report",
    )
    if source_report_sha256 is not None and recorded_source_sha256 != source_report_sha256:
        mismatches.append("source_cache_report_sha256 differs from supplied source report")
    if source_report is not None and report.get(
        "raw_content_manifest_sha256"
    ) != source_report.get("raw_content_manifest_sha256"):
        mismatches.append("raw_content_manifest_sha256 differs from supplied source report")
    if mismatches:
        raise ValueError("validation report differs from bundle: " + "; ".join(mismatches))

    marker_expected = {
        "schema": RELEASE_MARKER_SCHEMA,
        "full_validation_sha256": sha256_file(report_path),
        "portable_split_meta_sha256": expected["portable_split_meta_sha256"],
        "supervision_pairs_manifest_sha256": expected[
            "supervision_pairs_manifest_sha256"
        ],
        "source_cache_report_sha256": recorded_source_sha256,
        "configuration_fingerprint": expected["configuration_fingerprint"],
        "cache_count": expected_count,
    }
    if marker != marker_expected:
        raise ValueError("VALIDATED_SUCCESS identity differs from full_validation.json")


def main() -> None:
    args = parse_args()
    expected_count = int(args.expected_count)
    if expected_count <= 0:
        raise ValueError("--expected-count must be positive")
    if args.write_contract and not str(args.source_cache_report or "").strip():
        raise ValueError("--source-cache-report is required with --write-contract")

    bundle_root = Path(args.bundle_root).expanduser().resolve(strict=True)
    if not bundle_root.is_dir():
        raise NotADirectoryError(bundle_root)
    if args.write_contract:
        home = Path.home().resolve()
        if bundle_root == home or home in bundle_root.parents:
            raise ValueError(f"refusing to write validation contracts below home: {bundle_root}")

    started = time.time()
    validated = validate_layout(bundle_root, expected_count)
    source_report = None
    source_report_sha256 = None
    if str(args.source_cache_report or "").strip():
        source_path = Path(args.source_cache_report).expanduser().resolve(strict=True)
        source_report, source_report_sha256 = validate_source_cache_report(
            source_path,
            expected_count=expected_count,
            feature_summary=validated["feature_summary"],
        )

    report_path = bundle_root / "full_validation.json"
    marker_path = bundle_root / "VALIDATED_SUCCESS"
    if args.write_contract:
        assert source_report is not None and source_report_sha256 is not None
        report = build_release_report(
            expected_count=expected_count,
            validated=validated,
            source_report=source_report,
            source_report_sha256=source_report_sha256,
            elapsed_seconds=time.time() - started,
        )
        write_json_atomic(report_path, report)
        marker = build_success_marker(report_path, report)
        write_json_atomic(marker_path, marker)
        mode = "WROTE"
    else:
        verify_existing_contract(
            report_path=report_path,
            marker_path=marker_path,
            expected_count=expected_count,
            validated=validated,
            source_report=source_report,
            source_report_sha256=source_report_sha256,
        )
        mode = "VERIFIED"

    print(
        f"[BUNDLE-{mode}] samples={expected_count} "
        f"elapsed_seconds={time.time() - started:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
