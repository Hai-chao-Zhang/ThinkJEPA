#!/usr/bin/env python3

# ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model
# Copyright (c) 2024-2026 Northeastern University.
# Developed in NEU SMILE LAB by Haichao Zhang (https://zhanghaichao.xyz)
# and Yun Raymond Fu (https://www1.ece.neu.edu/~yunfu/).
# SPDX-style identifier: LicenseRef-ThinkJEPA-Attribution
# Original source: https://github.com/Hai-chao-Zhang/ThinkJEPA
# See the root LICENSE, NOTICE, CITATION.cff, and CITATION.bib for attribution and citation requirements.

import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cache_train.runtime_paths import (  # noqa: E402
    resolve_egodex_data_reference,
)


_QWEN_CACHE_NAME_RE = re.compile(
    r"^(?P<stem>.+?)_L\d+_nf\d+_res\d+_new\d+_s\d+of\d+$"
)
_EPISODE_CLIP_SUFFIX_RE = re.compile(
    r"(?:[-_](?:clip|chunk|part|segment|window)\d+|[-_]frames?\d+(?:[-_]\d+)?)$",
    re.IGNORECASE,
)
_GROUP_BY_CHOICES = ("auto", "take", "capture", "parent", "episode_stem")


@dataclass(frozen=True)
class VideoCachePair:
    video_path: str
    cache_path: str


def _normalized_relative_video_stem(video_path: str, data_root: str) -> str:
    """Return a root-independent POSIX path without the video suffix."""
    rel = os.path.relpath(os.path.abspath(video_path), os.path.abspath(data_root))
    rel = Path(os.path.normpath(rel)).as_posix()
    suffix = Path(rel).suffix
    return rel[: -len(suffix)] if suffix else rel


def _component_after(parts: tuple[str, ...], markers: set[str]) -> str | None:
    lowered = tuple(part.lower() for part in parts)
    for idx, part in enumerate(lowered[:-1]):
        if part in markers and parts[idx + 1] not in {"", ".", ".."}:
            return parts[idx + 1]
    return None


def _prefixed_component(parts: tuple[str, ...], prefix: str) -> str | None:
    prefix_re = re.compile(rf"^{re.escape(prefix)}(?:[-_].+|\d.*)$", re.IGNORECASE)
    for part in parts[:-1]:
        if prefix_re.match(part):
            return part
    return None


def _try_egoexo_capture(relative_stem: str) -> str | None:
    parts = tuple(Path(relative_stem).parts)
    return _component_after(parts, {"capture", "captures"}) or _prefixed_component(
        parts, "capture"
    )


def _try_egoexo_take(relative_stem: str) -> str | None:
    parts = tuple(Path(relative_stem).parts)
    take = _component_after(parts, {"take", "takes"}) or _prefixed_component(
        parts, "take"
    )
    if take:
        return take

    # A common EgoExo4D layout is
    # ``<take>/frame_aligned_videos/<camera>.mp4`` when data_root already
    # points at ``takes``.
    lowered = tuple(part.lower() for part in parts)
    for marker in ("frame_aligned_videos", "frame_aligned_video"):
        if marker in lowered:
            idx = lowered.index(marker)
            if idx > 0:
                return parts[idx - 1]
    if len(parts) > 1 and parts[0] not in {"", ".", ".."}:
        return parts[0]
    if parts:
        return parts[-1]
    return None


def _canonical_episode_stem(stem: str) -> str:
    """Collapse conventional clip suffixes while retaining the episode id."""
    previous = stem
    while True:
        current = _EPISODE_CLIP_SUFFIX_RE.sub("", previous)
        if current == previous:
            return current or stem
        previous = current


def _regex_group_key(relative_stem: str, group_regex: str) -> str:
    try:
        pattern = re.compile(group_regex)
    except re.error as exc:
        raise ValueError(f"invalid --group_regex {group_regex!r}: {exc}") from exc
    match = pattern.search(relative_stem)
    if match is None:
        raise ValueError(
            f"--group_regex {group_regex!r} did not match relative video path "
            f"{relative_stem!r}"
        )
    if "group" in pattern.groupindex:
        value = match.group("group")
    elif match.lastindex:
        value = match.group(1)
    else:
        value = match.group(0)
    value = (value or "").strip().strip("/\\")
    if not value:
        raise ValueError(
            f"--group_regex {group_regex!r} produced an empty group for "
            f"{relative_stem!r}"
        )
    return f"regex:{value}"


def infer_video_group_key(
    dataset: str,
    video_path: str,
    data_root: str,
    group_by: str = "auto",
    group_regex: str | None = None,
) -> str:
    """Infer a stable, root-independent leakage group for one source video.

    A custom regex overrides the dataset rule. Its named ``group`` capture, its
    first positional capture, or the full match (in that order) becomes the key.
    Callers processing a complete EgoExo4D set should resolve ``auto`` with
    :func:`resolve_group_by` first so every path uses one granularity.
    """
    if dataset not in {"egodex", "egoexo4d"}:
        raise ValueError(f"unsupported dataset for grouping: {dataset!r}")
    if group_by not in _GROUP_BY_CHOICES:
        raise ValueError(
            f"group_by must be one of {'|'.join(_GROUP_BY_CHOICES)}, got {group_by!r}"
        )

    relative_stem = _normalized_relative_video_stem(video_path, data_root)
    if group_regex:
        return _regex_group_key(relative_stem, group_regex)

    parts = tuple(Path(relative_stem).parts)
    parent = Path(relative_stem).parent.as_posix()
    episode_stem = _canonical_episode_stem(Path(relative_stem).name)

    if dataset == "egodex":
        effective = "parent" if group_by == "auto" else group_by
        if effective == "parent":
            # EgoDex is laid out as <task>/<episode>.mp4. Grouping by the
            # relative task directory is deliberately conservative. If a flat
            # custom export has no task directory, fall back to episode ids.
            return (
                f"egodex-task:{parent}"
                if parent not in {"", "."}
                else f"egodex-episode:{episode_stem}"
            )
        if effective == "episode_stem":
            episode_path = (
                f"{parent}/{episode_stem}"
                if parent not in {"", "."}
                else episode_stem
            )
            return f"egodex-episode:{episode_path}"
        if effective in {"take", "capture"}:
            raise ValueError(
                f"--group_by {effective} is only meaningful for egoexo4d; "
                "use auto, parent, episode_stem, or --group_regex for egodex"
            )

    effective = "take" if group_by == "auto" else group_by
    if effective == "capture":
        capture = _try_egoexo_capture(relative_stem)
        if capture is None:
            raise ValueError(
                "could not infer an EgoExo4D capture from relative path "
                f"{relative_stem!r}; choose --group_by take or provide "
                "--group_regex with a capture group"
            )
        return f"egoexo-capture:{capture}"
    if effective == "take":
        take = _try_egoexo_take(relative_stem)
        if take is None:
            raise ValueError(
                "could not infer an EgoExo4D take from relative path "
                f"{relative_stem!r}; provide --group_regex"
            )
        return f"egoexo-take:{take}"
    if effective == "parent":
        return (
            f"egoexo-parent:{parent}"
            if parent not in {"", "."}
            else f"egoexo-video:{episode_stem}"
        )
    if effective == "episode_stem":
        episode_path = (
            f"{parent}/{episode_stem}" if parent not in {"", "."} else episode_stem
        )
        return f"egoexo-video:{episode_path}"
    raise AssertionError(f"unhandled group_by={effective!r} for parts={parts!r}")


def resolve_group_by(
    dataset: str,
    pairs: list[VideoCachePair],
    data_root: str,
    requested: str,
    group_regex: str | None = None,
) -> str:
    """Resolve ``auto`` once for the full set to avoid mixed granularity."""
    if group_regex:
        return "regex"
    if requested != "auto":
        return requested
    if dataset == "egodex":
        return "parent"
    relative_stems = [
        _normalized_relative_video_stem(pair.video_path, data_root) for pair in pairs
    ]
    if relative_stems and all(_try_egoexo_capture(rel) for rel in relative_stems):
        return "capture"
    return "take"


def deduplicate_source_video_pairs(
    pairs: list[VideoCachePair],
) -> tuple[list[VideoCachePair], int]:
    """Keep one deterministic cache per physical source video."""
    unique: list[VideoCachePair] = []
    seen: set[str] = set()
    for pair in sorted(pairs, key=lambda p: (p.video_path, p.cache_path)):
        identity = os.path.normcase(os.path.realpath(os.path.abspath(pair.video_path)))
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(pair)
    return unique, len(pairs) - len(unique)


def _bucket_pairs_by_group(
    pairs: list[VideoCachePair], group_keys: list[str]
) -> dict[str, list[VideoCachePair]]:
    if len(pairs) != len(group_keys):
        raise ValueError("pairs and group_keys must have the same length")
    buckets: dict[str, list[VideoCachePair]] = {}
    for pair, group_key in zip(pairs, group_keys):
        buckets.setdefault(group_key, []).append(pair)
    return buckets


def select_group_aware_subset(
    pairs: list[VideoCachePair],
    group_keys: list[str],
    subset_size: int,
    split_seed: int,
) -> tuple[list[VideoCachePair], list[str]]:
    """Select exactly ``subset_size`` samples while maximizing group coverage."""
    if subset_size < 0 or subset_size > len(pairs):
        raise ValueError(
            f"subset_size must be in [0,{len(pairs)}], got {subset_size}"
        )
    buckets = _bucket_pairs_by_group(pairs, group_keys)
    rnd = random.Random(split_seed)
    ordered_groups = sorted(buckets)
    rnd.shuffle(ordered_groups)
    for group_key in ordered_groups:
        buckets[group_key] = sorted(
            buckets[group_key], key=lambda p: (p.video_path, p.cache_path)
        )
        rnd.shuffle(buckets[group_key])

    selected_pairs: list[VideoCachePair] = []
    selected_keys: list[str] = []
    offsets = {group_key: 0 for group_key in ordered_groups}
    while len(selected_pairs) < subset_size:
        made_progress = False
        for group_key in ordered_groups:
            offset = offsets[group_key]
            if offset >= len(buckets[group_key]):
                continue
            selected_pairs.append(buckets[group_key][offset])
            selected_keys.append(group_key)
            offsets[group_key] += 1
            made_progress = True
            if len(selected_pairs) == subset_size:
                break
        if not made_progress:
            raise AssertionError("group-aware subset selection exhausted early")
    return selected_pairs, selected_keys


def split_pairs_by_group(
    pairs: list[VideoCachePair],
    group_keys: list[str],
    train_ratio: float,
    split_seed: int,
) -> tuple[list[VideoCachePair], list[VideoCachePair], set[str], set[str]]:
    """Partition whole groups, choosing the shuffled prefix nearest the ratio."""
    buckets = _bucket_pairs_by_group(pairs, group_keys)
    if len(buckets) < 2:
        raise ValueError(
            "group-aware train/test splitting requires at least two selected "
            "groups; change grouping or subset_size"
        )
    rnd = random.Random(split_seed)
    ordered_groups = sorted(buckets)
    rnd.shuffle(ordered_groups)
    target_train = _compute_train_subset_end(len(pairs), train_ratio)
    cumulative = 0
    candidates: list[tuple[int, int]] = []
    for cut, group_key in enumerate(ordered_groups[:-1], start=1):
        cumulative += len(buckets[group_key])
        candidates.append((abs(cumulative - target_train), cut))
    _, best_cut = min(candidates)
    ordered_train_groups = ordered_groups[:best_cut]
    ordered_test_groups = ordered_groups[best_cut:]
    train_groups = set(ordered_train_groups)
    test_groups = set(ordered_test_groups)
    if train_groups & test_groups:
        raise AssertionError("internal error: a group crossed the train/test split")

    train_pairs = [pair for key in ordered_train_groups for pair in buckets[key]]
    test_pairs = [pair for key in ordered_test_groups for pair in buckets[key]]
    rnd.shuffle(train_pairs)
    rnd.shuffle(test_pairs)
    return train_pairs, test_pairs, train_groups, test_groups


def hash_group_keys(group_keys: set[str]) -> str:
    """Hash a group set using an unambiguous canonical JSON representation."""
    payload = json.dumps(
        sorted(group_keys), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_thinker_cache_stem(npz_path: str) -> str:
    base = os.path.splitext(os.path.basename(npz_path))[0]
    m = _QWEN_CACHE_NAME_RE.match(base)
    return m.group("stem") if m else base


def _build_thinker_cache_index(cache_root: str) -> dict[tuple[str, str], str]:
    index: dict[tuple[str, str], str] = {}
    files = sorted(glob.glob(os.path.join(cache_root, "**", "*.npz"), recursive=True))
    for p in files:
        rel = os.path.relpath(p, cache_root)
        rel_dir = os.path.normpath(os.path.dirname(rel))
        stem = _normalize_thinker_cache_stem(p)
        key = (rel_dir, stem)
        previous = index.get(key)
        if previous is not None and os.path.realpath(previous) != os.path.realpath(p):
            raise ValueError(
                f"ambiguous cache archives for relative key {key}: "
                f"{previous} and {p}; remove legacy/duplicate caches"
            )
        index[key] = p
    return index


def _list_egodex_video_cache_pairs(data_root: str, cache_root: str) -> list[VideoCachePair]:
    index = _build_thinker_cache_index(cache_root)
    out: list[VideoCachePair] = []
    h5_files = sorted(glob.glob(os.path.join(data_root, "**", "*.hdf5"), recursive=True))
    for h5_path in h5_files:
        rel = os.path.splitext(os.path.relpath(h5_path, data_root))[0]
        rel_dir = os.path.normpath(os.path.dirname(rel))
        stem = os.path.basename(rel)
        cache_path = index.get((rel_dir, stem))
        if cache_path is None:
            continue
        video_path = os.path.splitext(h5_path)[0] + ".mp4"
        if not os.path.isfile(video_path):
            continue
        out.append(VideoCachePair(video_path=video_path, cache_path=cache_path))
    return out


def _list_egoexo_video_cache_pairs(data_root: str, cache_root: str) -> list[VideoCachePair]:
    index = _build_thinker_cache_index(cache_root)
    out: list[VideoCachePair] = []
    mp4_files = sorted(glob.glob(os.path.join(data_root, "**", "*.mp4"), recursive=True))
    for video_path in mp4_files:
        rel = os.path.splitext(os.path.relpath(video_path, data_root))[0]
        rel_dir = os.path.normpath(os.path.dirname(rel))
        stem = os.path.basename(rel)
        cache_path = index.get((rel_dir, stem))
        if cache_path is None:
            continue
        out.append(VideoCachePair(video_path=video_path, cache_path=cache_path))
    return out


def _compute_train_subset_end(total: int, train_ratio: float) -> int:
    if total <= 1:
        return total
    a = int(total * train_ratio)
    return max(1, min(a, total - 1))


def _write_manifest_lines(path: str, lines: list[str]):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line)
            f.write("\n")


def _write_video_cache_pairs_tsv(path: str, pairs: list[VideoCachePair]):
    with open(path, "w", encoding="utf-8") as f:
        f.write("video_path\tcache_path\n")
        for p in pairs:
            f.write(f"{p.video_path}\t{p.cache_path}\n")


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_cache_split_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["egodex", "egoexo4d"], required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--cache_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--subset_size", type=int, default=2000)
    p.add_argument("--train_ratio", type=float, default=0.9)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument(
        "--group_by",
        "--group_key",
        dest="group_by",
        choices=_GROUP_BY_CHOICES,
        default="auto",
        help=(
            "leakage group inferred from the data-root-relative video path "
            "(default: auto; EgoDex=parent task, EgoExo4D=capture when all "
            "paths expose one, otherwise take)"
        ),
    )
    p.add_argument(
        "--group_regex",
        default=None,
        help=(
            "optional regex applied to each extensionless relative video path; "
            "named group 'group', first capture, or full match overrides --group_by"
        ),
    )
    return p.parse_args(argv)


def main():
    args = parse_cache_split_args()
    args.data_root = resolve_egodex_data_reference(args.data_root)
    args.cache_root = resolve_egodex_data_reference(args.cache_root)
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0,1), got {args.train_ratio}")
    if args.subset_size <= 0:
        raise ValueError(f"subset_size must be positive, got {args.subset_size}")

    if args.dataset == "egodex":
        discovered_pairs = _list_egodex_video_cache_pairs(
            args.data_root, args.cache_root
        )
    else:
        discovered_pairs = _list_egoexo_video_cache_pairs(
            args.data_root, args.cache_root
        )

    pairs, duplicate_source_video_pairs_dropped = deduplicate_source_video_pairs(
        discovered_pairs
    )

    if len(pairs) < args.subset_size:
        raise ValueError(
            f"dataset={args.dataset} only has {len(pairs)} matched video-cache pairs, "
            f"smaller than requested subset_size={args.subset_size}"
        )

    resolved_group_by = resolve_group_by(
        args.dataset,
        pairs,
        args.data_root,
        requested=args.group_by,
        group_regex=args.group_regex,
    )
    effective_group_by = (
        args.group_by if resolved_group_by == "regex" else resolved_group_by
    )
    all_group_keys = [
        infer_video_group_key(
            args.dataset,
            pair.video_path,
            args.data_root,
            group_by=effective_group_by,
            group_regex=args.group_regex,
        )
        for pair in pairs
    ]
    available_groups = set(all_group_keys)

    selected_pairs, selected_group_keys = select_group_aware_subset(
        pairs,
        all_group_keys,
        subset_size=args.subset_size,
        split_seed=args.split_seed,
    )
    train_pairs, test_pairs, train_groups, test_groups = split_pairs_by_group(
        selected_pairs,
        selected_group_keys,
        train_ratio=args.train_ratio,
        split_seed=args.split_seed,
    )

    selected_groups = set(selected_group_keys)
    group_intersection = train_groups & test_groups
    group_intersection_assertion_passed = not group_intersection
    if group_intersection:
        raise AssertionError(
            "group-aware split invariant failed: "
            f"{len(group_intersection)} groups crossed train/test"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    train_video = os.path.join(args.output_dir, "train_video.txt")
    test_video = os.path.join(args.output_dir, "test_video.txt")
    train_cache = os.path.join(args.output_dir, "train_cache.txt")
    test_cache = os.path.join(args.output_dir, "test_cache.txt")
    train_pairs_tsv = os.path.join(args.output_dir, "train_pairs.tsv")
    test_pairs_tsv = os.path.join(args.output_dir, "test_pairs.tsv")
    meta_json = os.path.join(args.output_dir, "meta.json")

    _write_manifest_lines(train_video, [p.video_path for p in train_pairs])
    _write_manifest_lines(test_video, [p.video_path for p in test_pairs])
    _write_manifest_lines(train_cache, [p.cache_path for p in train_pairs])
    _write_manifest_lines(test_cache, [p.cache_path for p in test_pairs])
    _write_video_cache_pairs_tsv(train_pairs_tsv, train_pairs)
    _write_video_cache_pairs_tsv(test_pairs_tsv, test_pairs)

    meta = {
        "dataset": args.dataset,
        "data_root": args.data_root,
        "cache_root": args.cache_root,
        "output_dir": args.output_dir,
        "subset_size": args.subset_size,
        "train_ratio": args.train_ratio,
        "split_seed": args.split_seed,
        "split_mode": "group_aware",
        "sample_level_split": False,
        "group_by_requested": args.group_by,
        "group_by_resolved": resolved_group_by,
        "group_regex": args.group_regex,
        "discovered_matched_pairs": len(discovered_pairs),
        "unique_matched_pairs": len(pairs),
        "duplicate_source_video_pairs_dropped": duplicate_source_video_pairs_dropped,
        "matched_pairs": len(selected_pairs),
        "train_count": len(train_pairs),
        "test_count": len(test_pairs),
        "available_group_count": len(available_groups),
        "selected_group_count": len(selected_groups),
        "train_group_count": len(train_groups),
        "test_group_count": len(test_groups),
        "group_hash_algorithm": "sha256-canonical-json-v1",
        "selected_group_hash": hash_group_keys(selected_groups),
        "train_group_hash": hash_group_keys(train_groups),
        "test_group_hash": hash_group_keys(test_groups),
        "group_intersection_count": len(group_intersection),
        "group_intersection_hash": hash_group_keys(group_intersection),
        "group_intersection_assertion_enforced": True,
        "group_intersection_assertion_passed": group_intersection_assertion_passed,
        "train_video_manifest": train_video,
        "train_video_manifest_sha256": _sha256_file(train_video),
        "test_video_manifest": test_video,
        "test_video_manifest_sha256": _sha256_file(test_video),
        "train_cache_manifest": train_cache,
        "train_cache_manifest_sha256": _sha256_file(train_cache),
        "test_cache_manifest": test_cache,
        "test_cache_manifest_sha256": _sha256_file(test_cache),
        "train_pairs_tsv": train_pairs_tsv,
        "test_pairs_tsv": test_pairs_tsv,
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
