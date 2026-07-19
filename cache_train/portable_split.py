"""Validate portable, content-addressed EgoDex supervision bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np


PORTABLE_SPLIT_SCHEMA = "thinkjepa.group_split.portable.v1"
PORTABLE_PATH_MODE = "bundle_relative"
PORTABLE_BUNDLE_SCHEMA = "thinkjepa.portable_hdf5.v1"
PORTABLE_BUNDLE_VERSION = 1


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_relative_posix_path(
    value: str | Path, *, expected_suffix: str | None = None
) -> str:
    """Return one unambiguous root-relative POSIX path or fail closed."""

    text = str(value)
    if not text or "\x00" in text or "\\" in text:
        raise ValueError(f"unsafe portable relative path: {text!r}")
    path = PurePosixPath(text)
    if path.is_absolute() or not path.parts or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ValueError(f"unsafe portable relative path: {text!r}")
    canonical = path.as_posix()
    if canonical != text:
        raise ValueError(
            f"portable path is not canonical POSIX form: {text!r} != {canonical!r}"
        )
    if expected_suffix is not None and path.suffix.lower() != expected_suffix.lower():
        raise ValueError(
            f"portable path {text!r} must end in {expected_suffix!r}"
        )
    return canonical


def _normalize_relative_directory(value: str | Path) -> str:
    text = normalize_relative_posix_path(value)
    if PurePosixPath(text).suffix:
        raise ValueError(f"bundle root role must be a directory: {text!r}")
    return text


def resolve_relative_file(
    root: str | Path, relative: str | Path, *, expected_suffix: str | None = None
) -> Path:
    relative_text = normalize_relative_posix_path(
        relative, expected_suffix=expected_suffix
    )
    root_real = Path(root).expanduser().resolve(strict=True)
    relative_parts = PurePosixPath(relative_text).parts
    unresolved = root_real
    for part in relative_parts:
        unresolved = unresolved / part
        if unresolved.is_symlink():
            raise ValueError(
                f"portable path must not traverse a symlink: {relative_text!r}"
            )
    candidate = unresolved.resolve(strict=True)
    try:
        candidate.relative_to(root_real)
    except ValueError as exc:
        raise ValueError(
            f"portable path resolves outside its explicit root: {relative_text!r}"
        ) from exc
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _read_manifest_lines(path: str | Path, *, expected_suffix: str) -> list[str]:
    lines = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise ValueError(f"portable split manifest is empty: {path}")
    normalized = [
        normalize_relative_posix_path(line, expected_suffix=expected_suffix)
        for line in lines
    ]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"portable split manifest contains duplicates: {path}")
    return normalized


def _canonical_json_line(value: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def _read_canonical_pairs(path: str | Path) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if not line:
                raise ValueError(f"blank portable pair record at line {line_number}")
            try:
                pair = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid portable pair JSON at line {line_number}: {exc}"
                ) from exc
            if not isinstance(pair, dict) or _canonical_json_line(pair) != line:
                raise ValueError(
                    f"portable pair line {line_number} is not canonical JSON"
                )
            pairs.append(pair)
    if not pairs:
        raise ValueError("portable supervision pair manifest is empty")
    return pairs


def _strip_bundle_role(
    value: str | Path, role: str, *, expected_suffix: str
) -> str:
    path_text = normalize_relative_posix_path(value, expected_suffix=expected_suffix)
    path_parts = PurePosixPath(path_text).parts
    role_parts = PurePosixPath(role).parts
    if path_parts[: len(role_parts)] != role_parts or len(path_parts) <= len(
        role_parts
    ):
        raise ValueError(f"path {path_text!r} is outside bundle role {role!r}")
    return PurePosixPath(*path_parts[len(role_parts) :]).as_posix()


def _validate_sha256(value: object, *, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"invalid SHA-256 for {label}")
    return digest


def _expected_causal_frame_indices(total_frames: int) -> tuple[list[int], list[int]]:
    if total_frames < 2:
        raise ValueError(f"invalid source_total_frames={total_frames}")
    midpoint = total_frames // 2
    past = np.linspace(0, midpoint - 1, num=32, dtype=np.int64).tolist()
    target = np.linspace(
        midpoint, total_frames - 1, num=32, dtype=np.int64
    ).tolist()
    return past, target


def _resolve_bundle_metadata_file(
    split_meta_path: str | Path, recorded_relative: str | Path, *, suffix: str
) -> Path:
    """Resolve paths recorded relative to the portable bundle root.

    The canonical layout is ``<bundle>/manifests/portable_v1/meta.json``.
    """

    meta_path = Path(split_meta_path).expanduser().resolve(strict=True)
    if meta_path.parent.name != "portable_v1" or meta_path.parent.parent.name != "manifests":
        raise ValueError(f"portable split metadata is outside its canonical layout: {meta_path}")
    bundle_root = meta_path.parent.parent.parent
    return resolve_relative_file(
        bundle_root, recorded_relative, expected_suffix=suffix
    )


def validate_portable_split_bundle(
    *,
    meta: Mapping[str, Any],
    split_meta_path: str | Path,
    train_manifest: str | Path,
    test_manifest: str | Path,
    supervision_root: str | Path,
    cache_root: str | Path,
    verify_supervision_content: bool = False,
) -> dict[str, Any]:
    """Validate a portable-v1 split and its separate HDF5 supervision root.

    The builder hashes every HDF5 byte before publishing the success marker.
    Validation checks the recorded identity, every relative association, file
    set, size, and read-only bit. Set ``verify_supervision_content=True`` to
    rehash all HDF5 files once; DDP workers should not each repeat that read.
    """

    if (
        meta.get("split_schema") != PORTABLE_SPLIT_SCHEMA
        or meta.get("portable_path_mode") != PORTABLE_PATH_MODE
        or meta.get("dataset") != "egodex"
    ):
        raise ValueError("unknown or unsafe portable split metadata contract")

    train_hash = sha256_file(train_manifest)
    test_hash = sha256_file(test_manifest)
    if train_hash != str(meta.get("train_cache_manifest_sha256", "")):
        raise ValueError("train cache manifest hash differs from portable metadata")
    if test_hash != str(meta.get("test_cache_manifest_sha256", "")):
        raise ValueError("test cache manifest hash differs from portable metadata")
    train_relpaths = _read_manifest_lines(train_manifest, expected_suffix=".npz")
    test_relpaths = _read_manifest_lines(test_manifest, expected_suffix=".npz")
    if set(train_relpaths) & set(test_relpaths):
        raise ValueError("portable train/test cache manifests overlap")
    if len(train_relpaths) != int(meta.get("train_count", -1)) or len(
        test_relpaths
    ) != int(meta.get("test_count", -1)):
        raise ValueError("portable split counts differ from metadata")

    pair_manifest = _resolve_bundle_metadata_file(
        split_meta_path,
        str(meta.get("supervision_pairs_manifest", "")),
        suffix=".jsonl",
    )
    pair_manifest_sha256 = sha256_file(pair_manifest)
    if pair_manifest_sha256 != str(
        meta.get("supervision_pairs_manifest_sha256", "")
    ):
        raise ValueError("portable supervision pair-manifest hash mismatch")

    meta_path = Path(split_meta_path).expanduser().resolve(strict=True)
    report_path = meta_path.parent / "bundle_validation.json"
    marker_path = meta_path.parent / "PORTABLE_VALIDATED_SUCCESS"
    if not report_path.is_file() or not marker_path.is_file():
        raise FileNotFoundError(
            "portable supervision bundle lacks validation report/success marker"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "ok"
        or report.get("schema") != PORTABLE_BUNDLE_SCHEMA
        or int(report.get("schema_version", -1)) != PORTABLE_BUNDLE_VERSION
        or int(report.get("feature_count", -1))
        != int(meta.get("supervision_file_count", -2))
        or int(report.get("supervision_count", -1))
        != int(meta.get("supervision_file_count", -2))
        or int(report.get("train_count", -1)) != len(train_relpaths)
        or int(report.get("test_count", -1)) != len(test_relpaths)
        or int(report.get("train_test_overlap_count", -1)) != 0
        or report.get("pairs_manifest_sha256") != pair_manifest_sha256
        or report.get("portable_meta_sha256") != sha256_file(meta_path)
    ):
        raise ValueError("portable validation report is inconsistent or incomplete")

    cache_role = _normalize_relative_directory(
        str(meta.get("bundle_cache_root", ""))
    )
    supervision_role = _normalize_relative_directory(
        str(meta.get("bundle_supervision_root", ""))
    )
    query_names = meta.get("query_tf_names")
    if not isinstance(query_names, list) or sha256_json(query_names) != str(
        meta.get("query_tf_names_sha256", "")
    ):
        raise ValueError("portable query-transform identity is invalid")

    expected_pair_keys = {
        "bundle_schema",
        "sample_id",
        "split",
        "group",
        "feature_relpath",
        "raw_video_relpath",
        "supervision_relpath",
        "source_video_relpath",
        "source_total_frames",
        "past_frame_indices",
        "target_frame_indices",
        "frame_indices_sha256",
        "feature_schema",
        "feature_extractor_version",
        "feature_config_fingerprint",
        "feature_size_bytes",
        "hdf5_sha256",
        "hdf5_size_bytes",
        "query_tf_names_sha256",
        "confidences_present",
    }
    train_set = set(train_relpaths)
    test_set = set(test_relpaths)
    pair_by_feature: dict[str, dict[str, Any]] = {}
    hdf5_hashes: list[str] = []
    pairs = _read_canonical_pairs(pair_manifest)
    for pair in pairs:
        if set(pair) != expected_pair_keys:
            raise ValueError("portable pair record has an unexpected schema")
        if pair.get("bundle_schema") != PORTABLE_BUNDLE_SCHEMA:
            raise ValueError("portable pair has the wrong bundle schema")
        sample_npz = normalize_relative_posix_path(
            f"{pair['sample_id']}.npz", expected_suffix=".npz"
        )
        feature_relpath = _strip_bundle_role(
            pair["feature_relpath"], cache_role, expected_suffix=".npz"
        )
        supervision_relpath = _strip_bundle_role(
            pair["supervision_relpath"],
            supervision_role,
            expected_suffix=".hdf5",
        )
        source_video_relpath = normalize_relative_posix_path(
            pair["source_video_relpath"], expected_suffix=".mp4"
        )
        if (
            feature_relpath != sample_npz
            or PurePosixPath(supervision_relpath).with_suffix(".npz").as_posix()
            != sample_npz
            or PurePosixPath(source_video_relpath).with_suffix(".npz").as_posix()
            != sample_npz
        ):
            raise ValueError("portable feature/HDF5/source sample identities differ")
        expected_split = (
            "train"
            if feature_relpath in train_set
            else "test" if feature_relpath in test_set else None
        )
        if pair.get("split") != expected_split:
            raise ValueError(f"portable pair split mismatch for {feature_relpath}")
        if feature_relpath in pair_by_feature:
            raise ValueError(f"duplicate portable pair for {feature_relpath}")
        if pair.get("feature_schema") != "thinkjepa.causal_split.v2":
            raise ValueError("portable pair references a non-schema-v2 feature")
        if pair.get("feature_config_fingerprint") != meta.get(
            "feature_config_fingerprint"
        ):
            raise ValueError("portable feature configuration identity mismatch")
        past, target = _expected_causal_frame_indices(
            int(pair["source_total_frames"])
        )
        if pair["past_frame_indices"] != past or pair["target_frame_indices"] != target:
            raise ValueError("portable pair frame provenance differs from causal sampling")
        if sha256_json(past + target) != pair.get("frame_indices_sha256"):
            raise ValueError("portable pair frame-index identity mismatch")
        if pair.get("query_tf_names_sha256") != meta.get("query_tf_names_sha256"):
            raise ValueError("portable pair query-transform identity mismatch")
        hdf5_hash = _validate_sha256(
            pair.get("hdf5_sha256"), label=supervision_relpath
        )
        if int(pair.get("hdf5_size_bytes", -1)) <= 0 or int(
            pair.get("feature_size_bytes", -1)
        ) <= 0:
            raise ValueError("portable pair contains an invalid file size")
        pair_by_feature[feature_relpath] = {
            **pair,
            "resolved_supervision_relpath": supervision_relpath,
        }
        hdf5_hashes.append(hdf5_hash)

    selected_features = train_set | test_set
    if set(pair_by_feature) != selected_features:
        raise ValueError("portable pair set differs from train/test manifests")
    if len(set(hdf5_hashes)) != len(hdf5_hashes):
        raise ValueError("portable bundle contains duplicate HDF5 content")
    if sha256_json(sorted(hdf5_hashes)) != str(
        meta.get("supervision_content_set_sha256", "")
    ):
        raise ValueError("portable HDF5 content-set identity mismatch")
    if len(pair_by_feature) != int(meta.get("supervision_file_count", -1)):
        raise ValueError("portable supervision count mismatch")

    cache_root_real = Path(cache_root).expanduser().resolve(strict=True)
    supervision_root_real = Path(supervision_root).expanduser().resolve(strict=True)
    expected_hdf5_relpaths: set[str] = set()
    for feature_relpath, pair in sorted(pair_by_feature.items()):
        feature_path = resolve_relative_file(
            cache_root_real, feature_relpath, expected_suffix=".npz"
        )
        if feature_path.stat().st_size != int(pair["feature_size_bytes"]):
            raise ValueError(f"portable feature size changed for {feature_relpath}")
        supervision_relpath = pair["resolved_supervision_relpath"]
        supervision_path = resolve_relative_file(
            supervision_root_real,
            supervision_relpath,
            expected_suffix=".hdf5",
        )
        expected_hdf5_relpaths.add(supervision_relpath)
        if supervision_path.stat().st_size != int(pair["hdf5_size_bytes"]):
            raise ValueError(f"portable HDF5 size changed for {supervision_relpath}")
        if supervision_path.stat().st_mode & 0o222:
            raise ValueError(
                "portable HDF5 is writable; materialize the HF revision into a "
                "dedicated local data root, then remove write bits before training "
                f"(for example: chmod a-w): {supervision_path}"
            )
        if verify_supervision_content and sha256_file(supervision_path) != pair[
            "hdf5_sha256"
        ]:
            raise ValueError(f"portable HDF5 content changed for {supervision_relpath}")

    actual_hdf5_relpaths = {
        path.relative_to(supervision_root_real).as_posix()
        for path in supervision_root_real.rglob("*.hdf5")
        if path.is_file()
    }
    if actual_hdf5_relpaths != expected_hdf5_relpaths:
        raise ValueError("explicit supervision root contains missing or unbound HDF5 files")

    return {
        "train_hash": train_hash,
        "test_hash": test_hash,
        "train_relpaths": train_relpaths,
        "test_relpaths": test_relpaths,
        "supervision_identity_sha256": pair_manifest_sha256,
        "supervision_file_count": len(pair_by_feature),
        "content_rehashed": bool(verify_supervision_content),
    }
