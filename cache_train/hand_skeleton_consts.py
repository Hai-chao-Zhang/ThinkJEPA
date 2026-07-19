# ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model
# Copyright (c) 2024-2026 Northeastern University.
# Developed in NEU SMILE LAB by Haichao Zhang (https://zhanghaichao.xyz)
# and Yun Raymond Fu (https://www1.ece.neu.edu/~yunfu/).
# SPDX-style identifier: LicenseRef-ThinkJEPA-Attribution
# Original source: https://github.com/Hai-chao-Zhang/ThinkJEPA
# See the root LICENSE, NOTICE, CITATION.cff, and CITATION.bib for attribution and citation requirements.

"""
Hand / joint constants and masks for egodex pipeline.
Keep this file import-only: no heavy side effects.
"""

from __future__ import annotations

import torch

# Pull canonical finger/joint names from your skeleton defs
from egodex.utils.skeleton_tfs import (
    LEFT_FINGERS,
    LEFT_INDEX,
    LEFT_LITTLE,
    LEFT_MIDDLE,
    LEFT_RING,
    LEFT_THUMB,
    RIGHT_FINGERS,
    RIGHT_INDEX,
    RIGHT_LITTLE,
    RIGHT_MIDDLE,
    RIGHT_RING,
    RIGHT_THUMB,
)

# -------------------------
# Names / ordering
# -------------------------
QUERY_TFS = (
    RIGHT_FINGERS
    + ["rightHand", "rightForearm"]
    + LEFT_FINGERS
    + ["leftHand", "leftForearm"]
)

tf2idx = {k: i for i, k in enumerate(QUERY_TFS)}
JOINT_NAMES = QUERY_TFS
J_TOTAL = len(JOINT_NAMES)

# -------------------------
# Left/right split & masks
# -------------------------
n_right = len(RIGHT_FINGERS) + 2  # rightHand + rightForearm
n_left = len(LEFT_FINGERS) + 2  # leftHand + leftForearm

RIGHT_JOINT_MASK = [True] * n_right + [False] * n_left
LEFT_JOINT_MASK = [False] * n_right + [True] * n_left

# Tensors (1,1,J,1) for easy broadcast with (B,T,J,C) shapes
RIGHT_JOINT_MASK_T = torch.tensor(RIGHT_JOINT_MASK, dtype=torch.float32).view(
    1, 1, -1, 1
)
LEFT_JOINT_MASK_T = torch.tensor(LEFT_JOINT_MASK, dtype=torch.float32).view(1, 1, -1, 1)

# -------------------------
# Finger group dicts
# -------------------------
right_dict = {
    "index": RIGHT_INDEX,
    "thumb": RIGHT_THUMB,
    "middle": RIGHT_MIDDLE,
    "ring": RIGHT_RING,
    "little": RIGHT_LITTLE,
}
left_dict = {
    "index": LEFT_INDEX,
    "thumb": LEFT_THUMB,
    "middle": LEFT_MIDDLE,
    "ring": LEFT_RING,
    "little": LEFT_LITTLE,
}

# -------------------------
# Image normalization defaults
# -------------------------
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# -------------------------
# Misc
# -------------------------
visualize_flag = True

# Useful indices
right_idx = tf2idx["rightMiddleFingerMetacarpal"]
left_idx = tf2idx["leftMiddleFingerMetacarpal"]

__all__ = [
    # names / indices
    "QUERY_TFS",
    "tf2idx",
    "JOINT_NAMES",
    "J_TOTAL",
    # masks
    "RIGHT_JOINT_MASK",
    "LEFT_JOINT_MASK",
    "RIGHT_JOINT_MASK_T",
    "LEFT_JOINT_MASK_T",
    # groups
    "right_dict",
    "left_dict",
    # imagenet stats
    "IMAGENET_DEFAULT_MEAN",
    "IMAGENET_DEFAULT_STD",
    # misc
    "visualize_flag",
    "right_idx",
    "left_idx",
    # raw lists (in case callers need them)
    "RIGHT_FINGERS",
    "LEFT_FINGERS",
    "RIGHT_INDEX",
    "RIGHT_THUMB",
    "RIGHT_MIDDLE",
    "RIGHT_RING",
    "RIGHT_LITTLE",
    "LEFT_INDEX",
    "LEFT_THUMB",
    "LEFT_MIDDLE",
    "LEFT_RING",
    "LEFT_LITTLE",
]
