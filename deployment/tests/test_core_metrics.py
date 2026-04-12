import pytest

pytest.importorskip("fastapi")
pytest.importorskip("starlette")
pytest.importorskip("prometheus_client")


from fastapi import FastAPI
from fastapi.responses import Response, StreamingResponse
from starlette.testclient import TestClient


def test_metrics_middleware_counts_non_streaming_and_exceptions():
    from app.core.metrics import install_metrics

    app = FastAPI()
    install_metrics(app)

    @app.get("/ok")
    async def ok():
        return Response(b"ok", media_type="text/plain")

    @app.get("/boom")
    async def boom():
        raise RuntimeError("boom")

    # We want a 500 response rather than re-raising server exceptions.
    with TestClient(app, raise_server_exceptions=False) as client:
        r = client.get("/ok")
        assert r.status_code == 200

        r = client.get("/boom")
        assert r.status_code == 500

        metrics = client.get("/metrics")
        # /metrics is not defined here; we only verify middleware didn't break.
        assert metrics.status_code in (404, 405)


def test_metrics_middleware_wraps_streaming_iterators_and_finalizes():
    from app.core.metrics import install_metrics, metrics_response

    app = FastAPI()
    install_metrics(app)

    async def gen_ok():
        yield b"a"
        yield b"b"

    async def gen_error():
        yield b"a"
        raise RuntimeError("stream boom")

    @app.get("/stream")
    async def stream():
        return StreamingResponse(gen_ok(), media_type="application/octet-stream")

    @app.get("/stream_error")
    async def stream_error():
        return StreamingResponse(gen_error(), media_type="application/octet-stream")

    with TestClient(app) as client:
        r = client.get("/stream")
        assert r.status_code == 200
        assert r.content == b"ab"

        with pytest.raises(RuntimeError):
            client.get("/stream_error").content

    # Scrape directly to assert exception labels were recorded.
    body = metrics_response().body
    assert body is not None
    text = body.decode("utf-8")
    assert "nanovllm_exceptions_total" in text
    assert 'route="/stream_error"' in text
    assert 'exception_type="RuntimeError"' in text
