"""Startup probe proving the research sandbox denies Internet sockets."""

from __future__ import annotations

import errno
import json
import socket
from pathlib import Path


def _process_security_status() -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        key, _separator, raw = line.partition(":")
        if key in {"NoNewPrivs", "Seccomp"}:
            values[key] = int(raw.strip())
    return values.get("NoNewPrivs", 0), values.get("Seccomp", 0)


def main() -> None:
    if not hasattr(socket, "AF_UNIX"):
        raise SystemExit("AF_UNIX is unavailable")
    try:
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except OSError as exc:
        if exc.errno != errno.EPERM:
            raise SystemExit(f"AF_INET failed with unexpected errno {exc.errno}") from exc
    else:
        raise SystemExit("AF_INET socket unexpectedly succeeded")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM):
        pass
    no_new_privs, seccomp_mode = _process_security_status()
    if no_new_privs != 1 or seccomp_mode != 2:
        raise SystemExit(
            f"sandbox kernel state invalid: no_new_privs={no_new_privs} seccomp={seccomp_mode}"
        )
    print(
        json.dumps(
            {
                "no_new_privs": True,
                "seccomp_filter": True,
                "af_inet_denied": True,
                "af_unix_allowed": True,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
