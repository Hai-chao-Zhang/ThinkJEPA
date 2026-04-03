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
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cache_train.hf_egodex import configure_huggingface_cache_dirs, resolve_egodex_data_reference


def parse_manifest_generation_args():
    p = argparse.ArgumentParser("Generate a fixed train/test split manifest from files under a root.")
    p.add_argument("--data_root", required=True, help="Root containing files to split.")
    p.add_argument("--output_dir", required=True, help="Directory to write train/test manifests.")
    p.add_argument(
        "--glob_pattern",
        type=str,
        default="*.hdf5",
        help="Recursive file glob under data_root, e.g. '*.hdf5' or '*.mp4'.",
    )
    p.add_argument("--train_ratio", type=float, default=0.9, help="Train split ratio in (0,1).")
    p.add_argument("--split_seed", type=int, default=42, help="Shuffle seed before splitting.")
    p.add_argument(
        "--subset_size",
        type=int,
        default=0,
        help="If >0, keep only the first N samples after shuffle, then split within that subset.",
    )
    p.add_argument(
        "--no_shuffle",
        action="store_true",
        help="Disable the pre-split deterministic shuffle.",
    )
    return p.parse_manifest_generation_args()


def list_manifest_source_files(data_root: str, glob_pattern: str) -> List[str]:
    return sorted(glob.glob(os.path.join(data_root, "**", glob_pattern), recursive=True))


def compute_train_manifest_boundary(total: int, train_ratio: float) -> int:
    if total <= 0:
        return 0
    if total == 1:
        return 1
    a = int(total * float(train_ratio))
    return max(1, min(a, total - 1))


def write_manifest_lines(path: str, lines: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line)
            f.write("\n")


def main():
    configure_huggingface_cache_dirs()
    args = parse_manifest_generation_args()
    args.data_root = resolve_egodex_data_reference(args.data_root)
    if not (0.0 < float(args.train_ratio) < 1.0):
        raise ValueError(f"--train_ratio must be in (0,1), got {args.train_ratio}")

    files = list_manifest_source_files(args.data_root, args.glob_pattern)
    if not args.no_shuffle and len(files) > 1:
        rnd = random.Random(int(args.split_seed))
        rnd.shuffle(files)
    subset_size = int(args.subset_size)
    if subset_size > 0:
        files = files[: min(subset_size, len(files))]

    train_end = compute_train_manifest_boundary(len(files), args.train_ratio)
    train_files = files[:train_end]
    test_files = files[train_end:]

    os.makedirs(args.output_dir, exist_ok=True)
    train_txt = os.path.join(args.output_dir, "train.txt")
    test_txt = os.path.join(args.output_dir, "test.txt")
    meta_json = os.path.join(args.output_dir, "meta.json")

    write_manifest_lines(train_txt, train_files)
    write_manifest_lines(test_txt, test_files)

    meta = {
        "data_root": args.data_root,
        "output_dir": args.output_dir,
        "glob_pattern": args.glob_pattern,
        "train_ratio": float(args.train_ratio),
        "split_seed": int(args.split_seed),
        "split_shuffle": not bool(args.no_shuffle),
        "subset_size": subset_size,
        "total_files": len(files),
        "train_count": len(train_files),
        "test_count": len(test_files),
        "train_manifest": train_txt,
        "test_manifest": test_txt,
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")

    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
