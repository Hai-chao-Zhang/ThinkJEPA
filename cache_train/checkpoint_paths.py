# ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model
# Copyright (c) 2024-2026 Northeastern University.
# Developed in NEU SMILE LAB by Haichao Zhang (https://zhanghaichao.xyz)
# and Yun Raymond Fu (https://www1.ece.neu.edu/~yunfu/).
# SPDX-style identifier: LicenseRef-ThinkJEPA-Attribution
# Original source: https://github.com/Hai-chao-Zhang/ThinkJEPA
# See the root LICENSE, NOTICE, CITATION.cff, and CITATION.bib for attribution and citation requirements.

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VJEPA2_ROOT = Path(
    os.environ.get("VJEPA2_ROOT", str(REPO_ROOT / "external" / "vjepa2"))
)


def _resolve_release_path(label: str, env_var: str, candidates: list[Path]) -> str:
    env_value = os.environ.get(env_var)
    if env_value:
        return env_value

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    joined_candidates = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"Could not resolve {label}. Set {env_var} or place the file at one of:\n"
        f"{joined_candidates}"
    )


def resolve_dense_jepa_checkpoint() -> str:
    return _resolve_release_path(
        label="the ThinkJEPA JEPA-branch ViT-L checkpoint",
        env_var="THINKJEPA_JEPA_VITL_PT",
        candidates=[
            VJEPA2_ROOT / "vitl.pt",
            REPO_ROOT / "vjepa2" / "vitl.pt",
            REPO_ROOT / "checkpoints" / "vitl.pt",
            REPO_ROOT / "checkpoints" / "thinkjepa_jepa_vitl.pt",
        ],
    )
