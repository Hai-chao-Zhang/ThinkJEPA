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

from cache_train.hf_egodex import configure_huggingface_cache_dirs, resolve_egodex_data_reference


_QWEN_CACHE_NAME_RE = re.compile(
    r"^(?P<stem>.+?)_L\d+_nf\d+_res\d+_new\d+_s\d+of\d+$"
)


@dataclass(frozen=True)
class VideoCachePair:
    video_path: str
    cache_path: str


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
        if key not in index:
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


def parse_cache_split_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["egodex", "egoexo4d"], required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--cache_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--subset_size", type=int, default=2000)
    p.add_argument("--train_ratio", type=float, default=0.9)
    p.add_argument("--split_seed", type=int, default=42)
    return p.parse_cache_split_args()


def main():
    configure_huggingface_cache_dirs()
    args = parse_cache_split_args()
    args.data_root = resolve_egodex_data_reference(args.data_root)
    args.cache_root = resolve_egodex_data_reference(args.cache_root)
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0,1), got {args.train_ratio}")

    if args.dataset == "egodex":
        pairs = _list_egodex_video_cache_pairs(args.data_root, args.cache_root)
    else:
        pairs = _list_egoexo_video_cache_pairs(args.data_root, args.cache_root)

    if len(pairs) < args.subset_size:
        raise ValueError(
            f"dataset={args.dataset} only has {len(pairs)} matched video-cache pairs, "
            f"smaller than requested subset_size={args.subset_size}"
        )

    rnd = random.Random(args.split_seed)
    rnd.shuffle(pairs)
    pairs = pairs[: args.subset_size]

    train_end = _compute_train_subset_end(len(pairs), args.train_ratio)
    train_pairs = pairs[:train_end]
    test_pairs = pairs[train_end:]

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
        "matched_pairs": len(pairs),
        "train_count": len(train_pairs),
        "test_count": len(test_pairs),
        "train_video_manifest": train_video,
        "test_video_manifest": test_video,
        "train_cache_manifest": train_cache,
        "test_cache_manifest": test_cache,
        "train_pairs_tsv": train_pairs_tsv,
        "test_pairs_tsv": test_pairs_tsv,
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
