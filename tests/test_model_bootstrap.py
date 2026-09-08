from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import pytest

import fireviewer_orchestrator.bootstrap as bootstrap
import fireviewer_vision_runtime.model_provisioning as provisioning
from fireviewer_vision_runtime.model_provisioning import CacheStatus


def test_auto_prefetch_disabled_fails_before_starting_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FW_HF_CACHE_ROOT", str(tmp_path / "cache" / "hub"))
    monkeypatch.setenv("FW_ROMA_ROOT", str(tmp_path / "roma"))
    monkeypatch.setenv("FW_AUTO_PREFETCH_MODELS", "false")

    with pytest.raises(RuntimeError, match="auto-prefetch is disabled"):
        bootstrap.ensure_model_cache()


def test_missing_cache_is_provisioned_once_and_returns_offline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    statuses = iter(
        (
            CacheStatus(("asr",)),
            CacheStatus(("asr",)),
            CacheStatus(()),
        )
    )
    provisioned: list[tuple[Path, Path]] = []
    monkeypatch.setenv("FW_HF_CACHE_ROOT", str(tmp_path / "cache" / "hub"))
    monkeypatch.setenv("FW_ROMA_ROOT", str(tmp_path / "roma"))
    monkeypatch.setenv("FW_AUTO_PREFETCH_MODELS", "true")
    monkeypatch.setattr(bootstrap, "cache_status", lambda *_args, **_kwargs: next(statuses))
    monkeypatch.setattr(bootstrap, "_volume_lock", lambda _path: nullcontext())
    monkeypatch.setattr(
        bootstrap,
        "provision_model_cache",
        lambda cache, roma: provisioned.append((cache, roma)),
    )

    bootstrap.ensure_model_cache()

    assert len(provisioned) == 1
    assert provisioned[0][0] == (tmp_path / "cache" / "hub").resolve()
    assert provisioned[0][1] == (tmp_path / "roma").resolve()
    assert bootstrap.os.environ["HF_HUB_OFFLINE"] == "1"
    assert bootstrap.os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_ready_cache_skips_lock_and_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bootstrap, "cache_status", lambda *_args, **_kwargs: CacheStatus(()))
    monkeypatch.setattr(
        bootstrap,
        "_volume_lock",
        lambda _path: pytest.fail("ready cache must not acquire the provisioning lock"),
    )
    monkeypatch.setattr(
        bootstrap,
        "provision_model_cache",
        lambda *_args, **_kwargs: pytest.fail("ready cache must not download models"),
    )

    bootstrap.ensure_model_cache()


def test_runtime_mode_selects_only_fixed_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FW_RUN_MODE", "pod_validation")
    assert bootstrap._runtime_module() == "fireviewer_orchestrator.pod_validation"

    monkeypatch.setenv("FW_RUN_MODE", "arbitrary.module")
    with pytest.raises(RuntimeError, match="unsupported FW_RUN_MODE"):
        bootstrap._runtime_module()


def test_runtime_dependencies_are_reported_without_installing_at_boot(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    versions = {"torch": "2.8.0", "transformers": "4.55.0", "flash-attn": "2.8.3"}
    monkeypatch.setattr(bootstrap, "version", versions.__getitem__)
    monkeypatch.setenv("FW_ATTENTION_IMPLEMENTATION", "flash_attention_2")

    bootstrap.report_runtime_dependencies()

    output = capsys.readouterr().out
    assert "torch=2.8.0" in output
    assert "transformers=4.55.0" in output
    assert "flash_attn=2.8.3" in output
    assert "attention=flash_attention_2" in output


def test_detector_ensemble_changes_the_persisted_model_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FW_ENABLE_FIRE_DETECTOR_ENSEMBLE", "false")
    without_detector = provisioning._selected_models(skip_qwen=False)
    assert "fire_detection" not in {spec.role for spec in without_detector}

    monkeypatch.setenv("FW_ENABLE_FIRE_DETECTOR_ENSEMBLE", "true")
    with_detector = provisioning._selected_models(skip_qwen=False)
    assert "fire_detection" in {spec.role for spec in with_detector}
    detectors = [spec for spec in with_detector if spec.role == "fire_detection"]
    assert len(detectors) == 2
    assert len(with_detector) == len(without_detector) + 2


def test_fixed_gid_mount_falls_back_without_weakening_other_permissions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_file = tmp_path / "model.safetensors"
    model_file.write_bytes(b"weights")
    chmod_calls: list[int] = []

    def reject_chgrp(*_args: object, **_kwargs: object) -> None:
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(bootstrap.os, "chown", reject_chgrp, raising=False)
    monkeypatch.setattr(
        bootstrap.os,
        "chmod",
        lambda _path, mode: chmod_calls.append(mode),
    )

    mounted_gid = bootstrap._secure_storage_path(
        model_file,
        model_gid=10001,
        directory=False,
    )

    assert mounted_gid == model_file.stat().st_gid
    assert chmod_calls == [0o640]


def test_model_storage_groups_are_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(bootstrap.MODEL_STORAGE_GIDS_ENV, "10001,0,10001")

    assert bootstrap._model_storage_gids() == (0, 10001)


def test_provisioner_reuses_only_marked_complete_snapshots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache" / "hub"
    roma_root = tmp_path / "roma"
    cache_root.mkdir(parents=True)
    first, *missing = provisioning.enabled_public_models()
    first_snapshot = provisioning._snapshot_path(cache_root, first)
    first_snapshot.mkdir(parents=True)
    (first_snapshot / "model.safetensors").write_bytes(b"weights")
    marker = provisioning._manifest_path(cache_root)
    marker.write_text(
        json.dumps(
            {
                "models": [
                    {"model_id": first.model_id, "revision": first.revision, "role": first.role}
                ]
            }
        ),
        encoding="utf-8",
    )
    downloaded: list[str] = []

    def fake_snapshot_download(*, repo_id: str, revision: str, **_kwargs: object) -> str:
        spec = next(model for model in missing if model.model_id == repo_id)
        assert spec.revision == revision
        snapshot = provisioning._snapshot_path(cache_root, spec)
        snapshot.mkdir(parents=True)
        (snapshot / "model.safetensors").write_bytes(b"weights")
        downloaded.append(repo_id)
        return str(snapshot)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(
        provisioning,
        "provision_roma_assets",
        lambda _root: {"source_revision": "pinned-test-revision"},
    )

    manifest = provisioning.provision_model_cache(cache_root, roma_root)

    assert downloaded == [spec.model_id for spec in missing]
    assert manifest["models"][0]["model_id"] == first.model_id


def test_metadata_only_snapshot_is_never_marked_ready(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FW_ENABLE_RTDETR_BASELINE", "false")
    cache_root = tmp_path / "cache" / "hub"
    roma_root = tmp_path / "roma"
    first = provisioning.enabled_public_models()[0]
    snapshot = provisioning._snapshot_path(cache_root, first)
    snapshot.mkdir(parents=True)
    (snapshot / "README.md").write_text("metadata only", encoding="utf-8")

    status = provisioning.cache_status(cache_root, roma_root)

    assert first.role in status.missing
