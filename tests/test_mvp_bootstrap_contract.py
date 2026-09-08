from __future__ import annotations

import pytest

import fireviewer_orchestrator.bootstrap as bootstrap


def test_bootstrap_rejects_a_different_stack_id(monkeypatch) -> None:
    monkeypatch.setattr(bootstrap, "version", lambda _distribution: "test")
    monkeypatch.setenv("FW_ATTENTION_IMPLEMENTATION", "flash_attention_2")
    monkeypatch.setenv("FW_MVP_STACK_ID", "some-other-stack")

    with pytest.raises(RuntimeError, match="does not match"):
        bootstrap.report_runtime_dependencies()
