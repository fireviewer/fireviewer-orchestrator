from __future__ import annotations

import json
import threading
from http.client import HTTPConnection

import fireviewer_orchestrator.pod_server as pod_server


def test_root_health_and_readiness_are_public_ready_endpoints() -> None:
    server = pod_server.PodHttpServer(("127.0.0.1", 0), auth_token="x" * 32)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for path in ("/", "/healthz", "/readyz"):
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            try:
                connection.request("GET", path)
                response = connection.getresponse()
                assert response.status == 200
                assert json.loads(response.read()) == {
                    "status": "ready",
                    "mode": "runpod-pod",
                }
            finally:
                connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_pod_queue_executes_one_wrapped_job(monkeypatch) -> None:
    completed = threading.Event()

    def handle(request):
        assert request == {"input": {"schema_version": "1.0", "batch_id": "batch-test"}}
        completed.set()
        return {
            "schema_version": "1.0",
            "batch_id": "batch-test",
            "status": "succeeded",
            "retryable": False,
            "model_runs": [],
            "items": [],
            "validation_errors": [],
            "boot_ms": 0,
        }

    monkeypatch.setattr(pod_server, "handle_job", handle)
    jobs = pod_server.PodJobQueue()
    job = jobs.submit({"input": {"schema_version": "1.0", "batch_id": "batch-test"}})

    assert completed.wait(timeout=2)
    jobs._pending.join()
    stored = jobs.get(job.job_id)
    assert stored is not None
    assert stored.status == "COMPLETED"
    assert stored.output is not None and stored.output["batch_id"] == "batch-test"


def test_queued_pod_job_can_be_cancelled_without_execution(monkeypatch) -> None:
    blocker_started = threading.Event()
    release_blocker = threading.Event()
    executed: list[str] = []

    def handle(request):
        batch_id = request["input"]["batch_id"]
        executed.append(batch_id)
        if batch_id == "blocker":
            blocker_started.set()
            assert release_blocker.wait(timeout=2)
        return {"batch_id": batch_id}

    monkeypatch.setattr(pod_server, "handle_job", handle)
    jobs = pod_server.PodJobQueue()
    blocker = jobs.submit({"input": {"batch_id": "blocker"}})
    assert blocker_started.wait(timeout=2)
    cancelled = jobs.submit({"input": {"batch_id": "cancelled"}})
    assert jobs.cancel(cancelled.job_id) is cancelled
    release_blocker.set()
    jobs._pending.join()

    assert jobs.get(blocker.job_id).status == "COMPLETED"
    assert jobs.get(cancelled.job_id).status == "CANCELLED"
    assert executed == ["blocker"]
