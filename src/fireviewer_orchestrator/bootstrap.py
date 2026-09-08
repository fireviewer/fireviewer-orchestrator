"""Cold-start bootstrap for the public FireWarning worker image."""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import BinaryIO, Protocol, cast

from fireviewer_orchestrator.bootstrap_status import BootstrapStatusServer, start_bootstrap_status_server
from fireviewer_vision_runtime.model_provisioning import cache_status, provision_model_cache
from fireviewer_contracts.mvp_stack import load_mvp_stack, mvp_stack_digest

DEFAULT_CACHE_ROOT = Path("/runpod-volume/huggingface-cache/hub")
DEFAULT_ROMA_ROOT = Path("/runpod-volume/firewarning-roma")
RUNTIME_MODULES = {
    "serverless": "fireviewer_orchestrator.handler",
    "pod": "fireviewer_orchestrator.pod_server",
    "pod_validation": "fireviewer_orchestrator.pod_validation",
}
_RESEARCH_CHILDREN: list[subprocess.Popen[bytes]] = []
MODEL_STORAGE_GIDS_ENV = "FW_MODEL_STORAGE_GIDS"


def report_runtime_dependencies() -> None:
    versions: dict[str, str] = {}
    for distribution in ("torch", "transformers", "flash-attn"):
        try:
            versions[distribution] = version(distribution)
        except PackageNotFoundError as exc:
            raise RuntimeError(f"required runtime dependency is absent: {distribution}") from exc
    attention = os.getenv("FW_ATTENTION_IMPLEMENTATION", "").strip()
    if attention != "flash_attention_2":
        raise RuntimeError("FW_ATTENTION_IMPLEMENTATION must be flash_attention_2")
    stack = load_mvp_stack()
    expected_stack_id = os.getenv("FW_MVP_STACK_ID", stack.stack_id).strip()
    if expected_stack_id != stack.stack_id:
        raise RuntimeError(
            f"configured MVP stack {expected_stack_id!r} does not match {stack.stack_id!r}"
        )
    print(
        "firewarning bootstrap: runtime dependencies ready "
        f"torch={versions['torch']} transformers={versions['transformers']} "
        f"flash_attn={versions['flash-attn']} attention={attention} "
        f"mvp_stack={stack.stack_id} mvp_digest={mvp_stack_digest(stack)}",
        flush=True,
    )


class _PasswdEntry(Protocol):
    pw_gid: int
    pw_uid: int
    pw_dir: str


class _PwdModule(Protocol):
    def getpwnam(self, name: str) -> _PasswdEntry: ...


class _GroupEntry(Protocol):
    gr_gid: int


class _GrpModule(Protocol):
    def getgrnam(self, name: str) -> _GroupEntry: ...


def _enabled(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _runtime_module() -> str:
    mode = os.getenv("FW_RUN_MODE", "serverless").strip().lower()
    try:
        return RUNTIME_MODULES[mode]
    except KeyError as exc:
        raise RuntimeError(f"unsupported FW_RUN_MODE: {mode}") from exc


@contextmanager
def _volume_lock(path: Path) -> Iterator[None]:
    """Serialize provisioning when several pods share the same model volume."""
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - the image runtime is Linux
        raise RuntimeError("the model bootstrap requires a Linux container") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    stream: BinaryIO
    with path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)  # type: ignore[attr-defined]
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]


def ensure_model_cache() -> None:
    cache_root = Path(os.getenv("FW_HF_CACHE_ROOT", str(DEFAULT_CACHE_ROOT))).resolve()
    roma_root = Path(os.getenv("FW_ROMA_ROOT", str(DEFAULT_ROMA_ROOT))).resolve()
    lock_path = Path(
        os.getenv(
            "FW_MODEL_PREFETCH_LOCK_PATH",
            str(cache_root.parent / ".firewarning-model-prefetch.lock"),
        )
    ).resolve()
    status = cache_status(cache_root, roma_root)
    if status.ready:
        print("firewarning bootstrap: mounted model cache is ready", flush=True)
        return
    if not _enabled(os.getenv("FW_AUTO_PREFETCH_MODELS"), default=True):
        missing = ", ".join(status.missing)
        raise RuntimeError(
            f"mounted model cache is incomplete and auto-prefetch is disabled: {missing}"
        )

    with _volume_lock(lock_path):
        # Another pod may have completed provisioning while this pod waited.
        status = cache_status(cache_root, roma_root)
        if status.ready:
            print("firewarning bootstrap: shared model cache became ready", flush=True)
            return
        print(
            "firewarning bootstrap: downloading missing pinned weights to mounted storage: "
            + ", ".join(status.missing),
            flush=True,
        )
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
        try:
            provision_model_cache(cache_root, roma_root)
        finally:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        final_status = cache_status(cache_root, roma_root)
        if not final_status.ready:
            missing = ", ".join(final_status.missing)
            raise RuntimeError(f"mounted model cache remains incomplete: {missing}")


def _model_storage_gids() -> tuple[int, ...]:
    values = os.getenv(MODEL_STORAGE_GIDS_ENV, "").split(",")
    return tuple(sorted({int(value) for value in values if value.strip()}))


def _assert_broker_cannot_read_models(storage_gids: set[int]) -> None:
    import pwd

    broker = cast(_PwdModule, pwd).getpwnam("broker")
    broker_gids = set(os.getgrouplist("broker", broker.pw_gid))  # type: ignore[attr-defined]
    overlap = storage_gids & broker_gids
    if overlap:
        raise RuntimeError(
            "mounted model storage uses a group available to the network broker: "
            + ",".join(str(gid) for gid in sorted(overlap))
        )


def _secure_storage_path(path: Path, *, model_gid: int, directory: bool) -> int:
    """Apply group-only access, tolerating mounts that reject chgrp.

    RunPod persistent volumes can expose a fixed host-side group and return
    EPERM for chown/chgrp. In that case the fixed group is granted only to the
    worker and researcher processes; the broker is checked separately.
    """
    with suppress(PermissionError):
        os.chown(path, -1, model_gid)  # type: ignore[attr-defined]
    os.chmod(path, 0o750 if directory else 0o640)
    return path.stat().st_gid


def secure_model_storage() -> tuple[int, ...]:
    """Keep weights readable by inference users and inaccessible to the broker user."""
    get_euid = getattr(os, "geteuid", None)
    if get_euid is None or get_euid() != 0:
        raise RuntimeError("model storage permissions require root before privilege drop")
    import grp

    cache_root = Path(os.getenv("FW_HF_CACHE_ROOT", str(DEFAULT_CACHE_ROOT))).resolve()
    roma_root = Path(os.getenv("FW_ROMA_ROOT", str(DEFAULT_ROMA_ROOT))).resolve()
    manifest = cache_root.parent / "firewarning-model-cache.json"
    fingerprint = sha256(manifest.read_bytes()).hexdigest()
    marker = cache_root.parent / ".firewarning-model-permissions-v2.json"
    if marker.is_file():
        try:
            payload = json.loads(marker.read_text(encoding="ascii"))
        except (OSError, ValueError, TypeError):
            payload = {}
        if payload.get("fingerprint") == fingerprint:
            cached_storage_gids = {int(gid) for gid in payload.get("storage_gids", [])}
            if cached_storage_gids:
                _assert_broker_cannot_read_models(cached_storage_gids)
                return tuple(sorted(cached_storage_gids))
    model_gid = cast(_GrpModule, grp).getgrnam("firewarning-model").gr_gid
    storage_gids: set[int] = set()
    for root in (cache_root.parent, roma_root):
        for directory, names, filenames in os.walk(root, followlinks=False):
            directory_path = Path(directory)
            storage_gids.add(
                _secure_storage_path(directory_path, model_gid=model_gid, directory=True)
            )
            for name in (*names, *filenames):
                path = directory_path / name
                if path.is_symlink():
                    continue
                storage_gids.add(
                    _secure_storage_path(path, model_gid=model_gid, directory=path.is_dir())
                )
    _assert_broker_cannot_read_models(storage_gids)
    marker.write_text(
        json.dumps(
            {"fingerprint": fingerprint, "storage_gids": sorted(storage_gids)},
            separators=(",", ":"),
        )
        + "\n",
        encoding="ascii",
    )
    os.chmod(marker, 0o600)
    print(
        "firewarning bootstrap: model storage secured "
        f"groups={','.join(str(gid) for gid in sorted(storage_gids))}",
        flush=True,
    )
    return tuple(sorted(storage_gids))


def _drop_runtime_privileges() -> None:
    """Drop the bootstrap's volume-write privileges before starting inference."""
    get_euid = getattr(os, "geteuid", None)
    if get_euid is None or get_euid() != 0:
        return
    try:
        import pwd
    except ImportError as exc:  # pragma: no cover - the image runtime is Linux
        raise RuntimeError("the model bootstrap requires a Linux user database") from exc
    username = os.getenv("FW_RUNTIME_USER", "worker")
    account = cast(_PwdModule, pwd).getpwnam(username)
    os.initgroups(username, account.pw_gid)  # type: ignore[attr-defined]
    getgroups = cast(Callable[[], list[int]], os.getgroups)  # type: ignore[attr-defined]
    setgroups = cast(Callable[[list[int]], None], os.setgroups)  # type: ignore[attr-defined]
    setgroups(sorted(set(getgroups()) | set(_model_storage_gids())))
    os.setgid(account.pw_gid)  # type: ignore[attr-defined]
    os.setuid(account.pw_uid)  # type: ignore[attr-defined]
    os.environ["HOME"] = account.pw_dir
    if get_euid() != account.pw_uid:
        raise RuntimeError(f"failed to drop privileges to {username}")


def _demote_to(username: str) -> Callable[[], None]:
    def demote() -> None:
        import grp
        import pwd

        account = cast(_PwdModule, pwd).getpwnam(username)
        shared_gid = cast(_GrpModule, grp).getgrnam("firewarning").gr_gid
        os.initgroups(username, shared_gid)  # type: ignore[attr-defined]
        if username == "researcher":
            getgroups = cast(Callable[[], list[int]], os.getgroups)  # type: ignore[attr-defined]
            setgroups = cast(Callable[[list[int]], None], os.setgroups)  # type: ignore[attr-defined]
            setgroups(sorted(set(getgroups()) | set(_model_storage_gids())))
        os.setgid(shared_gid)  # type: ignore[attr-defined]
        os.setuid(account.pw_uid)  # type: ignore[attr-defined]
        os.umask(0o077)

    return demote


def _service_environment(
    *,
    username: str,
    control_token: str,
    broker_socket: Path,
    service_socket: Path,
) -> dict[str, str]:
    import pwd

    account = cast(_PwdModule, pwd).getpwnam(username)
    passthrough = (
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "LD_LIBRARY_PATH",
    )
    environment = {key: os.environ[key] for key in passthrough if os.environ.get(key)}
    environment.update(
        {
            "HOME": account.pw_dir,
            "PATH": str(Path(sys.executable).parent),
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "FW_RESEARCH_BROKER_SOCKET": str(broker_socket),
            "FW_RESEARCH_SERVICE_SOCKET": str(service_socket),
            "FW_RESEARCH_BROKER_CONTROL_TOKEN": control_token,
        }
    )
    if username == "researcher":
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HOME": os.getenv("HF_HOME", "/runpod-volume/huggingface-cache"),
                "FW_HF_CACHE_ROOT": os.getenv(
                    "FW_HF_CACHE_ROOT", "/runpod-volume/huggingface-cache/hub"
                ),
                "FW_ATTENTION_IMPLEMENTATION": os.getenv(
                    "FW_ATTENTION_IMPLEMENTATION", "flash_attention_2"
                ),
                "FW_RESEARCH_MODEL_TIMEOUT_SECONDS": os.getenv(
                    "FW_RESEARCH_MODEL_TIMEOUT_SECONDS", "840"
                ),
            }
        )
    return environment


def _wait_for_socket(path: Path, process: subprocess.Popen[bytes], *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"research child exited before creating {path.name}")
        if path.exists():
            return
        time.sleep(0.05)
    raise RuntimeError(f"research child did not create {path.name}")


def start_research_runtime() -> None:
    get_euid = getattr(os, "geteuid", None)
    if get_euid is None or get_euid() != 0:
        raise RuntimeError("research isolation bootstrap requires root before privilege drop")
    run_directory = Path(os.getenv("FW_RESEARCH_RUN_DIRECTORY", "/run/firewarning"))
    broker_socket = Path(os.getenv("FW_RESEARCH_BROKER_SOCKET", str(run_directory / "broker.sock")))
    service_socket = Path(
        os.getenv("FW_RESEARCH_SERVICE_SOCKET", str(run_directory / "research.sock"))
    )
    sandbox_launcher = Path(
        os.getenv("FW_RESEARCH_SANDBOX_LAUNCHER", "/usr/local/bin/fw-research-sandbox")
    )
    if not sandbox_launcher.is_file():
        raise RuntimeError("research seccomp launcher is absent")

    import grp

    shared_gid = cast(_GrpModule, grp).getgrnam("firewarning").gr_gid
    run_directory.mkdir(parents=True, exist_ok=True)
    os.chown(run_directory, 0, shared_gid)  # type: ignore[attr-defined]
    os.chmod(run_directory, 0o770)  # noqa: S103 - private shared Unix sockets
    broker_socket.unlink(missing_ok=True)
    service_socket.unlink(missing_ok=True)
    control_token = secrets.token_urlsafe(48)
    broker_environment = _service_environment(
        username="broker",
        control_token=control_token,
        broker_socket=broker_socket,
        service_socket=service_socket,
    )
    research_environment = _service_environment(
        username="researcher",
        control_token=control_token,
        broker_socket=broker_socket,
        service_socket=service_socket,
    )
    broker = subprocess.Popen(
        [sys.executable, "-m", "fireviewer_evidence_ingestion.research_broker"],
        env=broker_environment,
        preexec_fn=_demote_to("broker"),
    )
    try:
        _wait_for_socket(broker_socket, broker, timeout=15)
        probe = subprocess.run(  # noqa: S603 - fixed sandbox executable and module
            [
                str(sandbox_launcher),
                sys.executable,
                "-m",
                "fireviewer_orchestrator.seccomp_probe",
            ],
            env=research_environment,
            preexec_fn=_demote_to("researcher"),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if probe.returncode != 0:
            raise RuntimeError(f"research seccomp probe failed: {probe.stderr.strip()[-500:]}")
        print(
            f"firewarning bootstrap: research sandbox verified {probe.stdout.strip()}",
            flush=True,
        )
        research = subprocess.Popen(  # noqa: S603 - fixed sandbox executable and module
            [
                str(sandbox_launcher),
                sys.executable,
                "-m",
                "fireviewer_evidence_ingestion.research_service",
            ],
            env=research_environment,
            preexec_fn=_demote_to("researcher"),
        )
        try:
            _wait_for_socket(service_socket, research, timeout=15)
        except Exception:
            research.terminate()
            research.wait(timeout=5)
            raise
    except Exception:
        broker.terminate()
        broker.wait(timeout=5)
        raise
    _RESEARCH_CHILDREN.extend((broker, research))


def main() -> None:
    status_server: BootstrapStatusServer | None = None
    try:
        runtime_module = _runtime_module()
        if runtime_module == RUNTIME_MODULES["pod"]:
            status_server = start_bootstrap_status_server(int(os.getenv("FW_POD_PORT", "8000")))
            status_server.status.update("validating_runtime")
        report_runtime_dependencies()
        if status_server is not None:
            status_server.status.update("provisioning_models")
        ensure_model_cache()
        if status_server is not None:
            status_server.status.update("securing_model_storage")
        storage_gids = secure_model_storage()
        os.environ[MODEL_STORAGE_GIDS_ENV] = ",".join(str(gid) for gid in storage_gids)
        if runtime_module != RUNTIME_MODULES["pod_validation"] and _enabled(
            os.getenv("FW_ENABLE_SOURCE_RESEARCH"), default=True
        ):
            if status_server is not None:
                status_server.status.update("starting_research_sandbox")
            start_research_runtime()

        # A fresh process guarantees that every inference library observes the
        # restored offline flags; the worker can never download a floating model.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if status_server is not None:
            status_server.status.update("launching_worker")
        _drop_runtime_privileges()
        # File descriptors are non-inheritable by default. The bootstrap health
        # socket therefore closes atomically at exec so the pod server can bind
        # the same port without an unobservable gap.
        os.execv(  # noqa: S606 - fixed executable and argv; shell invocation is intentional
            sys.executable,
            [sys.executable, "-m", runtime_module],
        )
    except Exception as exc:
        print(f"firewarning bootstrap failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        if status_server is not None:
            status_server.status.fail(exc)
            try:
                hold_seconds = int(os.getenv("FW_BOOTSTRAP_FAILURE_HOLD_SECONDS", "300"))
            except ValueError:
                hold_seconds = 300
            time.sleep(max(0, min(hold_seconds, 900)))
            status_server.close()
        raise SystemExit(78) from exc


if __name__ == "__main__":
    main()
