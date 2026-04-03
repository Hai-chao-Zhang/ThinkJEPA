# Release Audit

## Included

- Core ThinkJEPA training code under `cache_train/`
- Minimal EgoDex dataloading and visualization helpers strictly imported by the retained train/test paths
- Minimal public preprocessing and cache-extraction scripts:
  - `cache_train/generate_egodex_split_manifest.py`
  - `cache_train/build_video_cache_splits.py`
  - `cache_train/qwen3_cache_extractor.py`
  - `cache_train/qwen3_parallel_cache_extractor.py`
- Minimal public wrappers:
  - `scripts/train.sh`
  - `scripts/eval_main.sh`
- Bundled third-party V-JEPA2 source subset required by the retained ThinkJEPA training path under `vjepa2/`
- Release metadata and citation files at repository root
- `CITATION.cff` for software citation metadata
- Third-party notice files for retained EgoDex-derived helpers:
  - `egodex/LICENSE.txt`
  - `egodex/ACKNOWLEDGEMENTS.txt`
  - `egodex/utils/LICENSE.txt`
  - `egodex/utils/ACKNOWLEDGEMENTS.txt`

## Excluded

- Entire dataset trees and any sample data
- Cached features, prepared `.npz` archives, and Hugging Face cache directories
- All checkpoints / weights (`*.pt`, `*.pth`, `*.ckpt`)
- Internal train/test split manifests and split metadata from `cache_train/splits/`
- Logs, outputs, tensorboard artifacts, draft scripts, notebooks, temporary files, and unrelated experiment runners
- Unrelated vendored third-party source trees such as `LLaVA-*` and other non-minimal components
- Cluster-specific launch scripts and local environment heuristics from the original working tree

## Sanitized

- Removed hard-coded private filesystem paths from retained public defaults
- Replaced release-time output defaults with repo-relative `outputs/` locations
- Removed legacy internal EgoDex root paths from Hugging Face path resolution defaults
- Replaced internal launch scripts with minimal public wrappers using environment variables and placeholders
- Omitted manifest files that contained internal absolute paths and dataset inventories
- Translated retained source comments and user-facing inline notes to English
- Removed unused helper APIs that were not referenced by the retained train/eval path
- Added only the minimum public Qwen3-VL extraction and cache-manifest scripts needed by the documented preprocessing workflow
- Trimmed the bundled `vjepa2/` tree by removing unused EgoDex-specific local extensions, backup scripts, custom heads, and debug notes that are not required by the public ThinkJEPA reproduction path

## Skipped Because Potentially Sensitive

- `cache_train/splits/**` because the manifests and metadata expose internal absolute paths and dataset enumerations
- `cache_train/logs/**`, `results/**`, and other experiment artifacts because they may contain local paths or unpublished metrics
- Any external dependency source checkout not clearly approved for redistribution

## Manual Review Items

- Some retained source files still contain legacy upstream notice text in their bodies; these notices were preserved rather than removed.
- Retained `egodex/**` files carry Apple attribution in-file and should continue to ship with the accompanying third-party notice files above.
- Retained `vjepa2/**` files remain under their upstream third-party notice files `vjepa2/LICENSE` and `vjepa2/APACHE-LICENSE`.
- The linked Hugging Face cache and GitHub repositories should be manually reviewed for publication intent and access policy before announcement.
- If additional provenance review is required, inspect the retained files that include prior copyright headers before pushing this snapshot publicly.
- The root repository uses the custom `ThinkJEPA Attribution License (BSD-3-Clause-based, custom)`, while retained third-party subtrees continue to follow their own notice files.
