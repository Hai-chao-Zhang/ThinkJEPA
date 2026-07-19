from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cache_train.build_portable_hdf5_bundle import safe_relative_path
from cache_train.portable_split import (
    PORTABLE_BUNDLE_SCHEMA,
    PORTABLE_PATH_MODE,
    PORTABLE_SPLIT_SCHEMA,
    normalize_relative_posix_path,
    sha256_file,
    sha256_json,
    validate_portable_split_bundle,
)


def _canonical_line(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def _write_portable_fixture(root: Path):
    bundle = root / "bundle"
    cache_root = bundle / "cache"
    raw_root = bundle / "raw_videos"
    supervision_root = bundle / "supervision_hdf5"
    meta_dir = bundle / "manifests" / "portable_v1"
    meta_dir.mkdir(parents=True)

    query_names = ["rightHand", "leftHand"]
    query_hash = sha256_json(query_names)
    config_fingerprint = "a" * 64
    pairs = []
    train_relpaths = ["task/train.npz"]
    test_relpaths = ["task/test.npz"]
    for split, feature_relpath in (("train", train_relpaths[0]), ("test", test_relpaths[0])):
        sample_id = str(Path(feature_relpath).with_suffix(""))
        feature_path = cache_root / feature_relpath
        raw_path = raw_root / f"{sample_id}.mp4"
        supervision_relpath = f"{sample_id}.hdf5"
        supervision_path = supervision_root / supervision_relpath
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        supervision_path.parent.mkdir(parents=True, exist_ok=True)
        feature_path.write_bytes(f"feature-{split}".encode("ascii"))
        raw_path.write_bytes(f"raw-{split}".encode("ascii"))
        supervision_path.write_bytes(f"supervision-{split}".encode("ascii"))
        os.chmod(feature_path, 0o444)
        os.chmod(supervision_path, 0o444)
        past = np.linspace(0, 31, num=32, dtype=np.int64).tolist()
        target = np.linspace(32, 63, num=32, dtype=np.int64).tolist()
        pairs.append(
            {
                "bundle_schema": PORTABLE_BUNDLE_SCHEMA,
                "sample_id": sample_id,
                "split": split,
                "group": "task",
                "feature_relpath": f"cache/{feature_relpath}",
                "raw_video_relpath": f"raw_videos/{sample_id}.mp4",
                "supervision_relpath": f"supervision_hdf5/{supervision_relpath}",
                "source_video_relpath": f"{sample_id}.mp4",
                "source_total_frames": 64,
                "past_frame_indices": past,
                "target_frame_indices": target,
                "frame_indices_sha256": sha256_json(past + target),
                "feature_schema": "thinkjepa.causal_split.v2",
                "feature_extractor_version": "2.0.5",
                "feature_config_fingerprint": config_fingerprint,
                "feature_size_bytes": feature_path.stat().st_size,
                "hdf5_sha256": sha256_file(supervision_path),
                "hdf5_size_bytes": supervision_path.stat().st_size,
                "query_tf_names_sha256": query_hash,
                "confidences_present": True,
            }
        )

    train_text = "".join(f"{path}\n" for path in train_relpaths)
    test_text = "".join(f"{path}\n" for path in test_relpaths)
    pairs_text = "".join(_canonical_line(pair) for pair in pairs)
    train_path = meta_dir / "train_cache.txt"
    test_path = meta_dir / "test_cache.txt"
    pair_path = meta_dir / "pairs.jsonl"
    train_path.write_text(train_text, encoding="utf-8")
    test_path.write_text(test_text, encoding="utf-8")
    pair_path.write_text(pairs_text, encoding="utf-8")

    pair_hash = hashlib.sha256(pairs_text.encode("utf-8")).hexdigest()
    meta = {
        "dataset": "egodex",
        "split_schema": PORTABLE_SPLIT_SCHEMA,
        "portable_path_mode": PORTABLE_PATH_MODE,
        "split_mode": "group_aware",
        "sample_level_split": False,
        "group_intersection_count": 0,
        "group_intersection_assertion_passed": True,
        "train_count": 1,
        "test_count": 1,
        "bundle_cache_root": "cache",
        "bundle_supervision_root": "supervision_hdf5",
        "train_cache_manifest_sha256": hashlib.sha256(
            train_text.encode("utf-8")
        ).hexdigest(),
        "test_cache_manifest_sha256": hashlib.sha256(
            test_text.encode("utf-8")
        ).hexdigest(),
        "supervision_pairs_manifest": "manifests/portable_v1/pairs.jsonl",
        "supervision_pairs_manifest_sha256": pair_hash,
        "supervision_file_count": 2,
        "supervision_content_set_sha256": sha256_json(
            sorted(pair["hdf5_sha256"] for pair in pairs)
        ),
        "query_tf_names": query_names,
        "query_tf_names_sha256": query_hash,
        "feature_config_fingerprint": config_fingerprint,
    }
    meta_path = meta_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "status": "ok",
        "schema": PORTABLE_BUNDLE_SCHEMA,
        "schema_version": 1,
        "feature_count": 2,
        "supervision_count": 2,
        "train_count": 1,
        "test_count": 1,
        "train_test_overlap_count": 0,
        "pairs_manifest_sha256": pair_hash,
        "portable_meta_sha256": sha256_file(meta_path),
    }
    (meta_dir / "bundle_validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (meta_dir / "PORTABLE_VALIDATED_SUCCESS").write_text("", encoding="utf-8")
    root_report = {
        "status": "ok",
        "schema": "thinkjepa.causal_split.v2",
        "extractor_version": "2.0.5",
        "path_mode": "bundle_relative",
        "cache_root": "cache",
        "raw_root": "raw_videos",
        "cache_count": 2,
        "raw_count": 2,
        "raw_content_duplicate_count": 0,
        "cache_files_write_protected": True,
        "configuration_fingerprint": config_fingerprint,
        "portable_split_schema": PORTABLE_SPLIT_SCHEMA,
        "portable_split_meta_sha256": sha256_file(meta_path),
        "supervision_pairs_manifest_sha256": pair_hash,
    }
    root_report_path = bundle / "full_validation.json"
    root_report_path.write_text(
        json.dumps(root_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    root_marker = {
        "schema": "thinkjepa.cache_validation.portable.v1",
        "full_validation_sha256": sha256_file(root_report_path),
        "portable_split_meta_sha256": sha256_file(meta_path),
        "supervision_pairs_manifest_sha256": pair_hash,
    }
    (bundle / "VALIDATED_SUCCESS").write_text(
        json.dumps(root_marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return bundle, cache_root, supervision_root, meta_path, train_path, test_path


class PortableSupervisionTests(unittest.TestCase):
    def test_checkpoint_selection_best_policy(self):
        from cache_train.thinker_train import checkpoint_should_update

        self.assertTrue(checkpoint_should_update("best", 0.4, 0.5))
        self.assertFalse(checkpoint_should_update("best", 0.6, 0.5))
        self.assertTrue(checkpoint_should_update("validation", 0.4, 0.5))
        self.assertTrue(checkpoint_should_update("last", 0.6, 0.5))
        with self.assertRaisesRegex(ValueError, "unsupported checkpoint_selection"):
            checkpoint_should_update("unknown", 0.4, 0.5)

    def test_checkpoint_report_distinguishes_best_from_last(self):
        from cache_train.thinker_train import write_markdown_experiment_report

        args = SimpleNamespace(
            smoke_only=False,
            backbone="vjepa",
            predictor="thinkjepa",
            epochs=2,
            seed=42,
            trajmode="traj",
            past_T=32,
            future_T=32,
            checkpoint_selection="best",
        )
        logs = {
            "run_identity": {},
            "epochs": [{"epoch": 2, "val_avg_dist": 0.4, "val_final_dist": 0.5}],
            "best": {
                "epoch": 2,
                "ade": 0.4,
                "fde": 0.5,
                "selection_mode": "best",
                "selection_split": "validation",
                "selection_metric": "val_avg_dist",
                "ckpt": "ckpt_best.pt",
            },
        }
        with tempfile.TemporaryDirectory(prefix="thinkjepa-report-test-") as tmp:
            report = Path(tmp) / "report.md"
            write_markdown_experiment_report(report, args, logs, 10, 2)
            text = report.read_text(encoding="utf-8")
            self.assertIn("Validation-Selected Checkpoint", text)
            self.assertIn("selected_epoch_val_final_dist (FDE)", text)
            self.assertIn("No independent held-out test manifest", text)

            args.checkpoint_selection = "last"
            logs["best"].update(
                selection_mode="last",
                selection_split="final_epoch",
                selection_metric="none",
            )
            write_markdown_experiment_report(report, args, logs, 10, 2)
            text = report.read_text(encoding="utf-8")
            self.assertIn("Final-Epoch Checkpoint", text)
            self.assertIn("validation was not used for model selection", text)

    def test_bundle_contract_binds_source_cache_identity_without_host_paths(self):
        from cache_train.validate_release_bundle import (
            build_release_report,
            build_success_marker,
            validate_source_cache_report,
        )

        with tempfile.TemporaryDirectory(prefix="thinkjepa-bundle-contract-test-") as tmp:
            tmp_path = Path(tmp)
            meta_path = tmp_path / "meta.json"
            meta_path.write_text("{}\n", encoding="utf-8")
            feature_summary = {
                "configuration_fingerprint": "a" * 64,
                "qwen_checkpoint_sha": "b" * 40,
                "vjepa_checkpoint_sha256": "c" * 64,
                "vlm_old_length_histogram": {"1166": 2},
                "vlm_new_length_histogram": {"15": 2},
                "token_id_length_histogram": {"16": 2},
                "total_cache_bytes": 1234,
                "cache_file_set_sha256": "d" * 64,
            }
            source_report = {
                "status": "ok",
                "schema": "thinkjepa.causal_split.v2",
                "extractor_version": "2.0.5",
                "raw_count": 2,
                "cache_count": 2,
                "raw_content_duplicate_count": 0,
                "cache_files_write_protected": True,
                "raw_content_manifest_sha256": "e" * 64,
                "raw_root": "/mnt/private/raw",
                "cache_root": "/mnt/private/cache",
                **feature_summary,
            }
            source_path = tmp_path / "source.json"
            source_path.write_text(
                json.dumps(source_report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            loaded_source, source_sha = validate_source_cache_report(
                source_path,
                expected_count=2,
                feature_summary=feature_summary,
            )
            validated = {
                "meta": {
                    "supervision_pairs_manifest_sha256": "f" * 64,
                    "supervision_content_set_sha256": "1" * 64,
                },
                "meta_path": meta_path,
                "portable": {
                    "supervision_file_count": 2,
                    "content_rehashed": True,
                    "train_relpaths": ["task/train.npz"],
                    "test_relpaths": ["task/test.npz"],
                },
                "feature_summary": feature_summary,
            }
            report = build_release_report(
                expected_count=2,
                validated=validated,
                source_report=loaded_source,
                source_report_sha256=source_sha,
                elapsed_seconds=1.0,
            )
            report_path = tmp_path / "full_validation.json"
            report_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            marker = build_success_marker(report_path, report)
            contract_text = json.dumps({"report": report, "marker": marker})
            self.assertNotIn("/mnt/private", contract_text)
            self.assertEqual(
                marker["schema"], "thinkjepa.cache_validation.portable.v1"
            )

            source_report["cache_file_set_sha256"] = "9" * 64
            source_path.write_text(
                json.dumps(source_report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "cache_file_set_sha256 differs"
            ):
                validate_source_cache_report(
                    source_path,
                    expected_count=2,
                    feature_summary=feature_summary,
                )

    def test_portable_metrics_remove_host_paths_without_changing_values(self):
        from cache_train.thinker_train import build_training_logs

        output_dir = Path("/mnt/example/run")
        logs = {
            "run_identity": {"training_contract_sha256": "a" * 64},
            "best": {
                "epoch": 3,
                "ade": 0.5,
                "ckpt": "/mnt/example/run/ckpt_best.pt",
            },
            "epochs": [
                {
                    "epoch": 3,
                    "val_avg_dist": 0.5,
                    "ckpt": "/mnt/example/run/ckpt_latest.pt",
                }
            ],
        }
        portable = build_training_logs(logs, output_dir)
        self.assertEqual(portable["best"]["ckpt"], "ckpt_best.pt")
        self.assertEqual(portable["epochs"][0]["ckpt"], "ckpt_latest.pt")
        self.assertEqual(portable["best"]["ade"], 0.5)
        self.assertEqual(logs["best"]["ckpt"], "/mnt/example/run/ckpt_best.pt")

    def test_bundle_relative_cache_gate_relocates_and_rejects_legacy_report(self):
        from cache_train.thinker_train import validate_completed_cache_gate

        with tempfile.TemporaryDirectory(prefix="thinkjepa-root-gate-test-") as tmp:
            tmp_path = Path(tmp)
            bundle, *_ = _write_portable_fixture(tmp_path)
            relocated = tmp_path / "another-machine" / "iso"
            shutil.copytree(bundle, relocated, copy_function=shutil.copy2)
            args = SimpleNamespace(
                use_npz_cache=True,
                cache_dir=str(relocated / "cache"),
                split_meta=str(relocated / "manifests" / "portable_v1" / "meta.json"),
                cache_validation_report=str(relocated / "full_validation.json"),
                cache_validation_marker=str(relocated / "VALIDATED_SUCCESS"),
            )
            validate_completed_cache_gate(args)
            self.assertEqual(
                args.validated_cache_fingerprint,
                "a" * 64,
            )

            report_path = relocated / "full_validation.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report.pop("raw_root")
            shutil.rmtree(relocated / "raw_videos")
            report_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            marker_path = relocated / "VALIDATED_SUCCESS"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker["full_validation_sha256"] = sha256_file(report_path)
            marker_path.write_text(
                json.dumps(marker, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            validate_completed_cache_gate(args)

            report["path_mode"] = "absolute_legacy"
            report["cache_root"] = str(tmp_path / "unrelated-old-cache")
            report_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "requires bundle-relative cache validation"
            ):
                validate_completed_cache_gate(args)

    def test_portable_bundle_relocates_and_validation_detects_same_size_tamper(self):
        with tempfile.TemporaryDirectory(prefix="thinkjepa-portable-test-") as tmp:
            tmp_path = Path(tmp)
            _, cache_root, supervision_root, meta_path, train_path, test_path = (
                _write_portable_fixture(tmp_path)
            )
            relocated_cache = tmp_path / "another-machine" / "features"
            relocated_supervision = tmp_path / "another-machine" / "labels"
            shutil.copytree(cache_root, relocated_cache)
            shutil.copytree(
                supervision_root, relocated_supervision, copy_function=shutil.copy2
            )
            meta = json.loads(meta_path.read_text(encoding="utf-8"))

            result = validate_portable_split_bundle(
                meta=meta,
                split_meta_path=meta_path,
                train_manifest=train_path,
                test_manifest=test_path,
                supervision_root=relocated_supervision,
                cache_root=relocated_cache,
                verify_supervision_content=True,
            )
            self.assertEqual(result["supervision_file_count"], 2)
            self.assertIs(result["content_rehashed"], True)

            victim = relocated_supervision / "task" / "train.hdf5"
            original = victim.read_bytes()
            alias_target = relocated_supervision / "task" / "same-size-alias.bin"
            alias_target.write_bytes(b"x" * len(original))
            os.chmod(victim, 0o644)
            victim.unlink()
            victim.symlink_to(alias_target.name)
            with self.assertRaisesRegex(ValueError, "must not traverse a symlink"):
                validate_portable_split_bundle(
                    meta=meta,
                    split_meta_path=meta_path,
                    train_manifest=train_path,
                    test_manifest=test_path,
                    supervision_root=relocated_supervision,
                    cache_root=relocated_cache,
                    verify_supervision_content=False,
                )
            victim.unlink()
            victim.write_bytes(original)
            os.chmod(victim, 0o444)
            alias_target.unlink()

            os.chmod(victim, 0o644)
            victim.write_bytes(bytes([original[0] ^ 1]) + original[1:])
            os.chmod(victim, 0o444)
            with self.assertRaisesRegex(ValueError, "content changed"):
                validate_portable_split_bundle(
                    meta=meta,
                    split_meta_path=meta_path,
                    train_manifest=train_path,
                    test_manifest=test_path,
                    supervision_root=relocated_supervision,
                    cache_root=relocated_cache,
                    verify_supervision_content=True,
                )

    def test_portable_paths_reject_absolute_parent_and_backslash(self):
        for unsafe in (
            "/absolute/sample.npz",
            "../sample.npz",
            "task\\sample.npz",
        ):
            with self.subTest(unsafe=unsafe):
                with self.assertRaisesRegex(ValueError, "unsafe portable"):
                    normalize_relative_posix_path(unsafe, expected_suffix=".npz")

        for unsafe in (
            "/absolute/sample.mp4",
            "../sample.mp4",
            "task\\sample.mp4",
            "task/./sample.mp4",
            "task/sample.mp4\x00suffix",
        ):
            with self.subTest(builder_unsafe=unsafe):
                with self.assertRaisesRegex(ValueError, "unsafe relative"):
                    safe_relative_path(unsafe, suffix=".mp4")

    def test_schema_v2_hdf5_join_never_uses_cache_or_metadata_fallback(self):
        from egodex.trajectory_dataset import CameraGeometryLoadError, NpzCacheDataset

        with tempfile.TemporaryDirectory(prefix="thinkjepa-join-test-") as tmp:
            tmp_path = Path(tmp)
            supervision_root = tmp_path / "explicit-supervision"
            cache_root = tmp_path / "feature-cache"
            legacy_root = tmp_path / "legacy"
            expected = supervision_root / "task" / "sample.hdf5"
            cache_fallback = cache_root / "task" / "sample.hdf5"
            stale_hint = legacy_root / "sample.hdf5"
            for path in (expected, cache_fallback, stale_hint):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")

            dataset = object.__new__(NpzCacheDataset)
            dataset.supervision_root = os.path.realpath(supervision_root)
            dataset.cache_root = os.path.realpath(cache_root)
            dataset.data_roots = [os.path.realpath(legacy_root)]
            npz_path = str(cache_root / "task" / "sample.npz")

            resolved = dataset._resolve_hdf5_path_from_cache_archive(
                npz_path,
                hdf5_path_hint=str(stale_hint),
                video_path_hint=str(legacy_root / "sample.mp4"),
                source_video_relpath="task/sample.mp4",
            )
            self.assertEqual(resolved, os.path.realpath(expected))

            expected.unlink()
            self.assertIsNone(
                dataset._resolve_hdf5_path_from_cache_archive(
                    npz_path,
                    hdf5_path_hint=str(stale_hint),
                    video_path_hint=str(legacy_root / "sample.mp4"),
                    source_video_relpath="task/sample.mp4",
                )
            )
            with self.assertRaisesRegex(
                CameraGeometryLoadError, "unsafe source_video_relpath"
            ):
                dataset._resolve_hdf5_path_from_cache_archive(
                    npz_path,
                    hdf5_path_hint=str(stale_hint),
                    video_path_hint=None,
                    source_video_relpath="../sample.mp4",
                )
            for unsafe in (
                "task\\sample.mp4",
                "task/./sample.mp4",
                "task/sample.mp4\x00suffix",
            ):
                with self.subTest(loader_unsafe=unsafe):
                    with self.assertRaisesRegex(
                        CameraGeometryLoadError, "unsafe source_video_relpath"
                    ):
                        dataset._resolve_hdf5_path_from_cache_archive(
                            npz_path,
                            hdf5_path_hint=str(stale_hint),
                            video_path_hint=None,
                            source_video_relpath=unsafe,
                        )


if __name__ == "__main__":
    unittest.main()
