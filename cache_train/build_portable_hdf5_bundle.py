#!/usr/bin/env python3
"""Build a portable, immutable HDF5 supervision layer for schema-v2 caches.

The feature archives remain byte-for-byte unchanged.  This tool reads only
small provenance members from each feature NPZ and copies the corresponding
canonical HDF5 file into a separate ``supervision_hdf5`` tree.  Legacy mixed
cache supervision is never read or accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import time
from pathlib import Path, PurePosixPath

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from egodex.trajectory_dataset import (  # noqa: E402
    CAUSAL_CACHE_EXTRACTOR_VERSION,
    CAUSAL_CACHE_SAMPLING_STRATEGY,
    CAUSAL_CACHE_SCHEMA_NAME,
    CAUSAL_CACHE_SCHEMA_VERSION,
    CAUSAL_FEATURE_FORBIDDEN_KEYS,
    NPZ_TARGET_QUERY_TFS,
    _sample_dense_jepa_frame_indices,
)


BUNDLE_SCHEMA = "thinkjepa.portable_hdf5.v1"
BUNDLE_SCHEMA_VERSION = 1
COPY_CHUNK_BYTES = 16 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--hdf5-source-dir", required=True)
    parser.add_argument("--source-split-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-count", type=int, default=2000)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse an already copied destination only when its SHA256 matches the source.",
    )
    return parser.parse_args()


def scalar(archive: np.lib.npyio.NpzFile, key: str) -> object:
    if key not in archive.files:
        raise KeyError(f"missing feature provenance key {key!r}")
    value = np.asarray(archive[key])
    if value.size != 1:
        raise ValueError(f"{key} must be scalar, got shape={value.shape}")
    item = value.reshape(-1)[0]
    if hasattr(item, "item"):
        item = item.item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    return item


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def copy_and_hash_atomic(source: Path, destination: Path, *, resume: bool) -> str:
    source_hash = sha256_file(source)
    if destination.exists():
        if not resume:
            raise FileExistsError(f"destination already exists: {destination}")
        if not destination.is_file() or sha256_file(destination) != source_hash:
            raise ValueError(f"resume destination differs from source: {destination}")
        return source_hash

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f".{destination.name}.partial.{os.getpid()}")
    if tmp.exists():
        tmp.unlink()
    try:
        with source.open("rb") as src, tmp.open("xb") as dst:
            shutil.copyfileobj(src, dst, length=COPY_CHUNK_BYTES)
            dst.flush()
            os.fsync(dst.fileno())
        shutil.copystat(source, tmp, follow_symlinks=True)
        if sha256_file(tmp) != source_hash:
            raise IOError(f"copied HDF5 hash mismatch: {source} -> {tmp}")
        os.chmod(tmp, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        os.replace(tmp, destination)
    finally:
        if tmp.exists():
            tmp.unlink()
    return source_hash


def safe_relative_path(value: str, *, suffix: str) -> Path:
    if not value or "\\" in value or "\x00" in value:
        raise ValueError(f"unsafe relative path: {value!r}")
    posix_path = PurePosixPath(value)
    if (
        posix_path.is_absolute()
        or not posix_path.parts
        or any(part in {"", ".", ".."} for part in posix_path.parts)
        or posix_path.as_posix() != value
        or posix_path.suffix.lower() != suffix
    ):
        raise ValueError(f"unsafe relative path: {value!r}")
    return Path(*posix_path.parts)


def ensure_beneath(root: Path, relative: Path) -> Path:
    candidate = (root / relative).resolve()
    if os.path.commonpath([str(root), str(candidate)]) != str(root):
        raise ValueError(f"path escapes configured root: {relative}")
    return candidate


def manifest_relative_stems(path: Path, expected_suffix: str) -> list[str]:
    stems: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        source = Path(line)
        if source.suffix.lower() != expected_suffix:
            raise ValueError(f"unexpected manifest suffix at {path}: {line}")
        if source.is_absolute():
            parts = source.parts
            marker = "cache" if expected_suffix == ".npz" else "part2"
            indices = [index for index, part in enumerate(parts) if part == marker]
            if not indices:
                raise ValueError(f"cannot make manifest path portable: {line}")
            source = Path(*parts[indices[-1] + 1 :])
        source = safe_relative_path(source.as_posix(), suffix=expected_suffix)
        stems.append(source.with_suffix("").as_posix())
    if len(stems) != len(set(stems)):
        raise ValueError(f"manifest contains duplicates: {path}")
    return stems


def validate_hdf5(
    path: Path, *, expected_total_frames: int, expected_indices: np.ndarray
) -> tuple[bool, int]:
    with h5py.File(path, "r") as root:
        if "/transforms/camera" not in root or "/camera/intrinsic" not in root:
            raise KeyError(f"missing camera supervision in {path}")
        total_frames = int(root["/transforms/camera"].shape[0])
        if total_frames != expected_total_frames:
            raise ValueError(
                f"HDF5 frame count mismatch at {path}: "
                f"feature={expected_total_frames} hdf5={total_frames}"
            )
        sampled = np.asarray(_sample_dense_jepa_frame_indices(total_frames), dtype=np.int64)
        if not np.array_equal(sampled, expected_indices):
            raise ValueError(f"HDF5 sampling indices differ from feature provenance: {path}")

        selected = np.unique(sampled)
        camera = np.asarray(root["/transforms/camera"][selected], dtype=np.float32)
        intrinsic = np.asarray(root["/camera/intrinsic"][:], dtype=np.float32)
        if camera.shape != (selected.size, 4, 4) or not np.isfinite(camera).all():
            raise ValueError(f"invalid camera extrinsics in {path}: {camera.shape}")
        if intrinsic.shape not in {(3, 3), (1, 3, 3), (total_frames, 3, 3)}:
            raise ValueError(f"invalid camera intrinsics in {path}: {intrinsic.shape}")
        if not np.isfinite(intrinsic).all():
            raise ValueError(f"non-finite camera intrinsics in {path}")
        determinant = np.linalg.det(camera[:, :3, :3].astype(np.float64))
        if not np.isfinite(determinant).all() or np.any(np.abs(determinant) < 1e-6):
            raise ValueError(f"singular camera rotation in {path}")

        for tf_name in NPZ_TARGET_QUERY_TFS:
            key = f"/transforms/{tf_name}"
            if key not in root:
                raise KeyError(f"missing {key} in {path}")
            transforms = np.asarray(root[key][selected], dtype=np.float32)
            if transforms.shape != (selected.size, 4, 4) or not np.isfinite(transforms).all():
                raise ValueError(f"invalid {key} in {path}: {transforms.shape}")

        confidence_keys = [f"/confidences/{name}" for name in NPZ_TARGET_QUERY_TFS]
        confidence_present = [key in root for key in confidence_keys]
        if any(confidence_present) and not all(confidence_present):
            raise ValueError(f"partially missing joint confidences in {path}")
        return bool(all(confidence_present)), total_frames


def main() -> None:
    args = parse_args()
    feature_root = Path(args.feature_dir).expanduser().resolve()
    hdf5_root = Path(args.hdf5_source_dir).expanduser().resolve()
    split_root = Path(args.source_split_dir).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    home = Path.home().resolve()
    if output_root == home or home in output_root.parents:
        raise ValueError(f"refusing to write portable data below home: {output_root}")
    for root in (feature_root, hdf5_root, split_root):
        if not root.is_dir():
            raise FileNotFoundError(root)
    for source_root in (feature_root, hdf5_root):
        if (
            output_root == source_root
            or source_root in output_root.parents
            or output_root in source_root.parents
        ):
            raise ValueError("output root must be independent from source roots")

    feature_files = sorted(feature_root.rglob("*.npz"))
    expected_count = int(args.expected_count)
    if len(feature_files) != expected_count:
        raise ValueError(
            f"expected {expected_count} feature archives, found {len(feature_files)}"
        )

    train_stems = manifest_relative_stems(split_root / "train_cache.txt", ".npz")
    test_stems = manifest_relative_stems(split_root / "test_cache.txt", ".npz")
    if len(train_stems) + len(test_stems) != expected_count:
        raise ValueError("portable split count differs from expected bundle count")
    if set(train_stems) & set(test_stems):
        raise ValueError("train/test split overlap")
    split_by_stem = {stem: "train" for stem in train_stems}
    split_by_stem.update({stem: "test" for stem in test_stems})

    output_root.mkdir(parents=True, exist_ok=True)
    portable_manifest_dir = output_root / "manifests" / "portable_v1"
    success_marker = portable_manifest_dir / "PORTABLE_VALIDATED_SUCCESS"
    if success_marker.exists():
        raise FileExistsError(
            f"validated output already exists; refusing to mutate it: {success_marker}"
        )

    started = time.time()
    pairs: list[dict[str, object]] = []
    config_fingerprints: set[str] = set()
    hdf5_hashes: set[str] = set()
    confidence_missing = 0
    total_hdf5_bytes = 0
    query_names = list(NPZ_TARGET_QUERY_TFS)
    query_names_sha256 = sha256_json(query_names)

    for index, feature_path in enumerate(feature_files, start=1):
        rel_npz = feature_path.relative_to(feature_root)
        rel_stem = rel_npz.with_suffix("").as_posix()
        if split_by_stem.get(rel_stem) not in {"train", "test"}:
            raise ValueError(f"feature is not bound into the group-aware split: {rel_npz}")
        with np.load(feature_path, allow_pickle=False) as archive:
            forbidden = sorted(set(archive.files) & CAUSAL_FEATURE_FORBIDDEN_KEYS)
            if forbidden:
                raise ValueError(f"feature archive contains forbidden supervision: {feature_path}: {forbidden}")
            schema_name = str(scalar(archive, "cache_schema_name"))
            schema_version = int(scalar(archive, "cache_schema_version"))
            extractor_version = str(scalar(archive, "extractor_version"))
            sampling_strategy = str(scalar(archive, "sampling_strategy"))
            if (
                schema_name != CAUSAL_CACHE_SCHEMA_NAME
                or schema_version != CAUSAL_CACHE_SCHEMA_VERSION
                or extractor_version != CAUSAL_CACHE_EXTRACTOR_VERSION
                or sampling_strategy != CAUSAL_CACHE_SAMPLING_STRATEGY
            ):
                raise ValueError(f"unsupported feature provenance at {feature_path}")
            source_video_relpath = safe_relative_path(
                str(scalar(archive, "source_video_relpath")), suffix=".mp4"
            )
            if source_video_relpath.with_suffix(".npz") != rel_npz:
                raise ValueError(
                    f"feature path/source binding mismatch: {feature_path} -> {source_video_relpath}"
                )
            total_frames = int(scalar(archive, "source_total_frames"))
            past_indices = np.asarray(
                archive["vjepa_input_frame_indices"], dtype=np.int64
            ).reshape(-1)
            target_indices = np.asarray(
                archive["vjepa_target_frame_indices"], dtype=np.int64
            ).reshape(-1)
            if past_indices.size != 32 or target_indices.size != 32:
                raise ValueError(f"expected 32+32 frame indices at {feature_path}")
            if set(past_indices.tolist()) & set(target_indices.tolist()):
                raise ValueError(f"past/target frame overlap at {feature_path}")
            frame_indices = np.concatenate([past_indices, target_indices])
            config_fingerprint = str(scalar(archive, "cache_config_fingerprint"))
            config_fingerprints.add(config_fingerprint)

        source_hdf5_relpath = source_video_relpath.with_suffix(".hdf5")
        source_hdf5 = ensure_beneath(hdf5_root, source_hdf5_relpath)
        if not source_hdf5.is_file():
            raise FileNotFoundError(source_hdf5)
        confidence_present, checked_total = validate_hdf5(
            source_hdf5,
            expected_total_frames=total_frames,
            expected_indices=frame_indices,
        )
        if checked_total != total_frames:
            raise AssertionError("internal frame count mismatch")
        if not confidence_present:
            confidence_missing += 1

        destination_relpath = Path("supervision_hdf5") / source_hdf5_relpath
        destination = output_root / destination_relpath
        hdf5_sha256 = copy_and_hash_atomic(
            source_hdf5, destination, resume=bool(args.resume)
        )
        if hdf5_sha256 in hdf5_hashes:
            raise ValueError(f"duplicate HDF5 content detected: {source_hdf5}")
        hdf5_hashes.add(hdf5_sha256)
        hdf5_size = int(destination.stat().st_size)
        total_hdf5_bytes += hdf5_size

        pairs.append(
            {
                "bundle_schema": BUNDLE_SCHEMA,
                "sample_id": rel_stem,
                "split": split_by_stem[rel_stem],
                "group": source_video_relpath.parent.as_posix(),
                "feature_relpath": (Path("cache") / rel_npz).as_posix(),
                "raw_video_relpath": (
                    Path("raw_videos") / source_video_relpath
                ).as_posix(),
                "supervision_relpath": destination_relpath.as_posix(),
                "source_video_relpath": source_video_relpath.as_posix(),
                "source_total_frames": total_frames,
                "past_frame_indices": past_indices.tolist(),
                "target_frame_indices": target_indices.tolist(),
                "frame_indices_sha256": sha256_json(frame_indices.tolist()),
                "feature_schema": schema_name,
                "feature_extractor_version": extractor_version,
                "feature_config_fingerprint": config_fingerprint,
                "feature_size_bytes": int(feature_path.stat().st_size),
                "hdf5_sha256": hdf5_sha256,
                "hdf5_size_bytes": hdf5_size,
                "query_tf_names_sha256": query_names_sha256,
                "confidences_present": confidence_present,
            }
        )
        if index % 100 == 0 or index == expected_count:
            print(f"[PORTABLE-BUILD] {index}/{expected_count}", flush=True)

    if len(config_fingerprints) != 1:
        raise ValueError(f"mixed feature configurations: {sorted(config_fingerprints)}")
    if len(hdf5_hashes) != expected_count:
        raise ValueError("HDF5 content is not one-to-one")

    pairs_text = "".join(
        json.dumps(pair, sort_keys=True, separators=(",", ":")) + "\n"
        for pair in pairs
    )
    train_lines = "".join(f"{stem}.npz\n" for stem in train_stems)
    test_lines = "".join(f"{stem}.npz\n" for stem in test_stems)
    write_text_atomic(portable_manifest_dir / "pairs.jsonl", pairs_text)
    write_text_atomic(portable_manifest_dir / "train_cache.txt", train_lines)
    write_text_atomic(portable_manifest_dir / "test_cache.txt", test_lines)

    source_meta_path = split_root / "meta.json"
    source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
    portable_meta = {
        key: value
        for key, value in source_meta.items()
        if key
        not in {
            "cache_root",
            "data_root",
            "output_dir",
            "train_cache_manifest",
            "test_cache_manifest",
            "train_video_manifest",
            "test_video_manifest",
            "train_pairs_tsv",
            "test_pairs_tsv",
        }
    }
    portable_meta.update(
        {
            "split_schema": "thinkjepa.group_split.portable.v1",
            "portable_path_mode": "bundle_relative",
            "bundle_cache_root": "cache",
            "bundle_supervision_root": "supervision_hdf5",
            "train_cache_manifest": "manifests/portable_v1/train_cache.txt",
            "test_cache_manifest": "manifests/portable_v1/test_cache.txt",
            "train_cache_manifest_sha256": hashlib.sha256(
                train_lines.encode("utf-8")
            ).hexdigest(),
            "test_cache_manifest_sha256": hashlib.sha256(
                test_lines.encode("utf-8")
            ).hexdigest(),
            "source_split_meta_sha256": sha256_file(source_meta_path),
            "supervision_pairs_manifest": "manifests/portable_v1/pairs.jsonl",
            "supervision_pairs_manifest_sha256": hashlib.sha256(
                pairs_text.encode("utf-8")
            ).hexdigest(),
            "supervision_file_count": expected_count,
            "supervision_total_bytes": total_hdf5_bytes,
            "supervision_content_set_sha256": sha256_json(sorted(hdf5_hashes)),
            "query_tf_names": query_names,
            "query_tf_names_sha256": query_names_sha256,
            "confidence_complete_count": expected_count - confidence_missing,
            "confidence_missing_count": confidence_missing,
            "feature_config_fingerprint": next(iter(config_fingerprints)),
        }
    )
    meta_text = json.dumps(portable_meta, indent=2, sort_keys=True) + "\n"
    write_text_atomic(portable_manifest_dir / "meta.json", meta_text)

    report = {
        "status": "ok",
        "schema": BUNDLE_SCHEMA,
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "feature_schema": CAUSAL_CACHE_SCHEMA_NAME,
        "feature_count": expected_count,
        "supervision_count": expected_count,
        "supervision_unique_sha256_count": len(hdf5_hashes),
        "supervision_total_bytes": total_hdf5_bytes,
        "confidence_complete_count": expected_count - confidence_missing,
        "confidence_missing_count": confidence_missing,
        "train_count": len(train_stems),
        "test_count": len(test_stems),
        "train_test_overlap_count": 0,
        "pairs_manifest_sha256": hashlib.sha256(pairs_text.encode("utf-8")).hexdigest(),
        "portable_meta_sha256": hashlib.sha256(meta_text.encode("utf-8")).hexdigest(),
        "elapsed_seconds": time.time() - started,
        "validated_unix": time.time(),
    }
    report_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    write_text_atomic(portable_manifest_dir / "bundle_validation.json", report_text)
    write_text_atomic(success_marker, "")
    print(
        "[PORTABLE-PASS] "
        f"samples={expected_count} bytes={total_hdf5_bytes} "
        f"confidence_missing={confidence_missing} output={output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
