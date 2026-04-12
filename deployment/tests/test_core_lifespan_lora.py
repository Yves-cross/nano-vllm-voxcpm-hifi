from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("starlette")
pytest.importorskip("prometheus_client")


from starlette.testclient import TestClient


def test_lifespan_loads_lora_when_configured(monkeypatch, tmp_path: Path):
    # Configure LoRA startup via env so create_app() enables the LoRA branch.
    monkeypatch.setenv("NANOVLLM_LORA_URI", "file:///dummy")
    monkeypatch.setenv("NANOVLLM_LORA_ID", "test-lora")

    import app.core.lifespan as lifespan
    from app.services.lora_resolver import ResolvedArtifact

    # Avoid touching network/filesystem beyond tmp_path.
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()

    monkeypatch.setattr(
        lifespan,
        "resolve_lora_uri",
        lambda uri, cache_dir, expected_sha256: ResolvedArtifact(
            local_path=tmp_path / "artifact", cache_key="x"
        ),
    )
    monkeypatch.setattr(lifespan, "normalize_lora_checkpoint_path", lambda p: ckpt_dir)
    monkeypatch.setattr(lifespan, "load_lora_config_from_checkpoint", lambda p: None)

    class FakeServerPoolWithLora:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs
            self._stopped = False
            self._lora_loaded = False

        async def wait_for_ready(self):
            return None

        async def stop(self):
            self._stopped = True

        async def load_lora(self, path: str):
            self._lora_loaded = True

        async def set_lora_enabled(self, enabled: bool):
            return None

    monkeypatch.setattr(lifespan, "SERVER_FACTORY", FakeServerPoolWithLora)

    from app.main import create_app

    app = create_app()
    with TestClient(app) as client:
        assert client.app.state.ready is True
        assert client.app.state.lora["loaded"] is True
        assert getattr(client.app.state.server, "_lora_loaded", False) is True
        # We passed a LoRAConfig instance into the server constructor.
        assert client.app.state.server.kwargs.get("lora_config") is not None

    # On shutdown, lifespan should clear state.server.
    assert not hasattr(app.state, "server")
