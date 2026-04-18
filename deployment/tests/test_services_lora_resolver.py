import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path
from zipfile import ZipFile

import pytest

DEPLOYMENT_DIR = Path(__file__).resolve().parents[1]
if str(DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_DIR))


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def test_cache_key_depends_on_sha256():
    from app.services.lora_resolver import _cache_key

    k1 = _cache_key("file:///tmp/a", None)
    k2 = _cache_key("file:///tmp/a", None)
    assert k1 == k2

    k3 = _cache_key("file:///tmp/a", "abcd")
    k4 = _cache_key("file:///tmp/a", "ABCD")
    assert k3 == k4
    assert k3 != k1


def test_extract_if_archive_rejects_zipslip(tmp_path: Path):
    from app.services.lora_resolver import _extract_if_archive

    zpath = tmp_path / "bad.zip"
    with ZipFile(zpath, "w") as zf:
        zf.writestr("../evil.txt", "pwn")

    with pytest.raises(RuntimeError, match="Refusing to extract zip member"):
        _extract_if_archive(zpath, tmp_path / "out")


def test_extract_if_archive_rejects_tarslip(tmp_path: Path):
    from app.services.lora_resolver import _extract_if_archive

    tpath = tmp_path / "bad.tar"
    with tarfile.open(tpath, "w") as tf:
        info = tarfile.TarInfo(name="../evil.txt")
        payload = b"pwn"
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))

    with pytest.raises(RuntimeError, match="Refusing to extract tar member"):
        _extract_if_archive(tpath, tmp_path / "out")


def test_ensure_empty_dir_removes_existing(tmp_path: Path):
    from app.services.lora_resolver import _ensure_empty_dir

    d = tmp_path / "d"
    d.mkdir()
    (d / "x.txt").write_text("x", encoding="utf-8")
    assert any(d.iterdir())

    _ensure_empty_dir(d)
    assert d.exists()
    assert list(d.iterdir()) == []


def test_normalize_lora_checkpoint_path(tmp_path: Path):
    from app.services.lora_resolver import normalize_lora_checkpoint_path

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "lora_weights.safetensors").write_bytes(b"weights")

    assert normalize_lora_checkpoint_path(ckpt) == ckpt
    assert normalize_lora_checkpoint_path(ckpt / "lora_weights.safetensors") == ckpt

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="LoRA weights not found"):
        normalize_lora_checkpoint_path(empty)


def test_load_lora_config_from_checkpoint(tmp_path: Path):
    from app.services.lora_resolver import load_lora_config_from_checkpoint

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    assert load_lora_config_from_checkpoint(ckpt) is None

    (ckpt / "lora_config.json").write_text(
        json.dumps({"base_model": "ignored", "lora_config": {"r": 8, "alpha": 4.0}}),
        encoding="utf-8",
    )
    cfg = load_lora_config_from_checkpoint(ckpt)
    assert cfg is not None
    assert cfg.r == 8
    assert cfg.alpha == 4.0

    (ckpt / "lora_config.json").write_text(json.dumps({"base_model": "ignored"}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing 'lora_config'"):
        load_lora_config_from_checkpoint(ckpt)


def test_resolve_lora_uri_file_and_cache(tmp_path: Path):
    from app.services.lora_resolver import resolve_lora_uri

    src = tmp_path / "src"
    src.mkdir()
    weights = b"weights-v1"
    (src / "lora_weights.safetensors").write_bytes(weights)
    expected = _sha256_bytes(weights)

    r1 = resolve_lora_uri(uri=src.as_uri(), cache_dir=str(tmp_path / "cache"), expected_sha256=expected)
    assert r1.local_path.exists()

    # Cache fast-path should still work even if the source changes.
    (src / "lora_weights.safetensors").write_bytes(b"weights-v2")
    r2 = resolve_lora_uri(uri=src.as_uri(), cache_dir=str(tmp_path / "cache"), expected_sha256=expected)
    assert r2.cache_key == r1.cache_key
    assert r2.local_path == r1.local_path


def test_resolve_lora_uri_file_sha256_mismatch(tmp_path: Path):
    from app.services.lora_resolver import resolve_lora_uri

    src = tmp_path / "src"
    src.mkdir()
    (src / "lora_weights.safetensors").write_bytes(b"weights")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        resolve_lora_uri(
            uri=src.as_uri(),
            cache_dir=str(tmp_path / "cache"),
            expected_sha256="0" * 64,
        )


def test_resolve_lora_uri_https_zip_extracted(tmp_path: Path, monkeypatch):
    from app.services.lora_resolver import resolve_lora_uri

    zip_src = tmp_path / "artifact.zip"
    weights = b"weights"
    with ZipFile(zip_src, "w") as zf:
        zf.writestr("lora_weights.safetensors", weights)
        zf.writestr("lora_config.json", json.dumps({"lora_config": {"r": 4}}))

    def fake_urlretrieve(url: str, filename: Path):
        Path(filename).write_bytes(zip_src.read_bytes())
        return (str(filename), None)

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)

    r = resolve_lora_uri(
        uri="https://example.com/lora.zip",
        cache_dir=str(tmp_path / "cache"),
        expected_sha256=_sha256_bytes(weights),
    )
    assert r.local_path.is_dir()
    assert (r.local_path / "lora_weights.safetensors").read_bytes() == weights


def test_resolve_lora_uri_unsupported_scheme(tmp_path: Path):
    from app.services.lora_resolver import resolve_lora_uri

    with pytest.raises(ValueError, match="Unsupported LoRA URI scheme"):
        resolve_lora_uri(
            uri="ftp://example.com/x",
            cache_dir=str(tmp_path / "cache"),
            expected_sha256=None,
        )


def test_resolve_lora_uri_cache_fast_path_backward_compat(tmp_path: Path):
    from app.services.lora_resolver import _cache_key, resolve_lora_uri

    uri = "file:///tmp/does-not-matter"
    cache_dir = tmp_path / "cache"

    key = _cache_key(uri, None)
    dst_root = cache_dir / "lora" / key
    (dst_root / "artifact").mkdir(parents=True)
    (dst_root / ".resolved").write_text("ok", encoding="utf-8")

    r = resolve_lora_uri(uri=uri, cache_dir=str(cache_dir), expected_sha256=None)
    assert r.cache_key == key
    assert r.local_path == (dst_root / "artifact").resolve()


def test_resolve_lora_uri_cache_fast_path_empty_resolved_path_raises(tmp_path: Path):
    from app.services.lora_resolver import _cache_key, resolve_lora_uri

    uri = "file:///tmp/does-not-matter"
    cache_dir = tmp_path / "cache"

    key = _cache_key(uri, None)
    dst_root = cache_dir / "lora" / key
    dst_root.mkdir(parents=True)
    (dst_root / ".resolved").write_text("ok", encoding="utf-8")
    (dst_root / ".resolved_path").write_text("\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"empty \.resolved_path"):
        resolve_lora_uri(uri=uri, cache_dir=str(cache_dir), expected_sha256=None)
