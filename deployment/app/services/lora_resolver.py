from __future__ import annotations

import json
import hashlib
import os
import shutil
import tarfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from nanovllm_voxcpm.models.voxcpm.config import LoRAConfig


@dataclass(frozen=True)
class ResolvedArtifact:
    local_path: Path
    cache_key: str


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_empty_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _cache_key(uri: str, sha256: str | None) -> str:
    h = hashlib.sha256()
    h.update(uri.encode("utf-8"))
    if sha256:
        h.update(b"\n")
        h.update(sha256.lower().encode("utf-8"))
    return h.hexdigest()


def _extract_if_archive(src_file: Path, dst_dir: Path) -> Path:
    name = src_file.name.lower()
    if name.endswith(".zip"):
        _ensure_empty_dir(dst_dir)
        with zipfile.ZipFile(src_file) as zf:
            for zip_member in zf.infolist():
                # Avoid ZipSlip.
                out_path = (dst_dir / zip_member.filename).resolve()
                if not str(out_path).startswith(str(dst_dir.resolve()) + os.sep):
                    raise RuntimeError(f"Refusing to extract zip member outside target dir: {zip_member.filename}")
            zf.extractall(dst_dir)
        return dst_dir

    if name.endswith(".tar") or name.endswith(".tar.gz") or name.endswith(".tgz"):
        _ensure_empty_dir(dst_dir)
        with tarfile.open(src_file, "r:*") as tf:
            dst_resolved = dst_dir.resolve()
            for tar_member in tf.getmembers():
                out_path = (dst_dir / tar_member.name).resolve()
                if not str(out_path).startswith(str(dst_resolved) + os.sep):
                    raise RuntimeError(f"Refusing to extract tar member outside target dir: {tar_member.name}")
            tf.extractall(dst_dir)
        return dst_dir

    return src_file


def _find_lora_weights_file(root: Path) -> Path | None:
    """Locate lora_weights.safetensors / lora_weights.ckpt under a directory."""

    cand = root / "lora_weights.safetensors"
    if cand.exists():
        return cand
    cand = root / "lora_weights.ckpt"
    if cand.exists():
        return cand

    return None


def normalize_lora_checkpoint_path(resolved_path: Path) -> Path:
    """Return the directory that should be passed as lora checkpoint path."""

    if resolved_path.is_file():
        return resolved_path.parent

    weights = _find_lora_weights_file(resolved_path)
    if weights is None:
        raise FileNotFoundError(
            f"LoRA weights not found under {resolved_path} (expected lora_weights.safetensors or lora_weights.ckpt)"
        )
    return weights.parent


def load_lora_config_from_checkpoint(
    ckpt_dir: Path,
) -> LoRAConfig | None:
    """Parse lora_config.json if present.

    Expected (per VoxCPM docs):
      - lora_config.json with keys: base_model, lora_config
    """

    cfg_path = ckpt_dir / "lora_config.json"
    if not cfg_path.exists():
        return None

    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    lora_cfg_obj = data.get("lora_config")
    if not isinstance(lora_cfg_obj, dict):
        raise ValueError("Invalid lora_config.json: missing 'lora_config' object")
    return LoRAConfig(**lora_cfg_obj)


def resolve_lora_uri(*, uri: str, cache_dir: str, expected_sha256: str | None) -> ResolvedArtifact:
    """Resolve a LoRA artifact URI to a local path.

    Supported schemes:
    - file://... (file or directory)
    - http(s)://... (file; .zip/.tar(.gz) auto-extracted)
    - s3://bucket/key (file or prefix; requires boto3)
    - hf://repo_id@revision?path=... (requires huggingface_hub)
    """

    cache_root = Path(os.path.expanduser(cache_dir))
    cache_root.mkdir(parents=True, exist_ok=True)

    key = _cache_key(uri, expected_sha256)
    dst_root = cache_root / "lora" / key
    done_marker = dst_root / ".resolved"
    resolved_path_marker = dst_root / ".resolved_path"

    # Fast path: already resolved.
    if done_marker.exists():
        if resolved_path_marker.exists():
            rel_str = resolved_path_marker.read_text(encoding="utf-8").strip()
            if not rel_str:
                raise RuntimeError("Invalid resolver cache: empty .resolved_path")
            return ResolvedArtifact(local_path=(dst_root / rel_str).resolve(), cache_key=key)
        # Backward compatibility: older cache entries.
        return ResolvedArtifact(local_path=(dst_root / "artifact").resolve(), cache_key=key)

    parsed = urllib.parse.urlparse(uri)
    scheme = parsed.scheme

    tmp_root = dst_root / ".tmp"
    artifact_root = dst_root / "artifact"

    _ensure_empty_dir(tmp_root)
    _ensure_empty_dir(artifact_root)

    try:
        if scheme == "file":
            src = Path(urllib.request.url2pathname(parsed.path))
            if not src.exists():
                raise FileNotFoundError(f"file URI not found: {src}")
            if src.is_dir():
                shutil.copytree(src, artifact_root, dirs_exist_ok=True)
                resolved_path: Path = artifact_root
            else:
                dst_file = artifact_root / src.name
                shutil.copy2(src, dst_file)
                resolved_path = dst_file

        elif scheme in ("http", "https"):
            filename = Path(parsed.path).name or "lora_artifact"
            dl_path = artifact_root / filename
            urllib.request.urlretrieve(uri, dl_path)
            resolved_path = _extract_if_archive(dl_path, artifact_root / "extracted")

        elif scheme == "s3":
            try:
                import boto3
            except Exception as e:  # pragma: no cover
                raise RuntimeError("boto3 is required for s3:// URIs") from e

            bucket = parsed.netloc
            key_prefix = parsed.path.lstrip("/")
            if not bucket or not key_prefix:
                raise ValueError(f"Invalid s3 URI: {uri}")

            s3 = boto3.client("s3")
            if key_prefix.endswith("/"):
                # Prefix download.
                paginator = s3.get_paginator("list_objects_v2")
                for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
                    for obj in page.get("Contents", []) or []:
                        obj_key = obj.get("Key")
                        if not obj_key or obj_key.endswith("/"):
                            continue
                        rel_path = Path(obj_key[len(key_prefix) :])
                        out_path = artifact_root / rel_path
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        s3.download_file(bucket, obj_key, str(out_path))
                resolved_path = artifact_root
            else:
                out_path = artifact_root / Path(key_prefix).name
                s3.download_file(bucket, key_prefix, str(out_path))
                resolved_path = _extract_if_archive(out_path, artifact_root / "extracted")

        elif scheme == "hf":
            # Format: hf://repo_id@revision?path=relative/path
            try:
                from huggingface_hub import snapshot_download
            except Exception as e:  # pragma: no cover
                raise RuntimeError("huggingface_hub is required for hf:// URIs") from e

            repo_spec = (parsed.netloc + parsed.path).lstrip("/")
            if not repo_spec:
                raise ValueError(f"Invalid hf URI: {uri}")

            if "@" in repo_spec:
                repo_id, revision = repo_spec.split("@", 1)
            else:
                repo_id, revision = repo_spec, None

            qs = urllib.parse.parse_qs(parsed.query)
            subpath = (qs.get("path") or [""])[0]

            allow_patterns = None
            if subpath:
                allow_patterns = [subpath, f"{subpath}/**"]

            local_repo_dir = Path(
                snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    local_dir=str(artifact_root / "repo"),
                    allow_patterns=allow_patterns,
                )
            )
            resolved_path = local_repo_dir / subpath if subpath else local_repo_dir

        else:
            raise ValueError(f"Unsupported LoRA URI scheme: {scheme!r}")

        if expected_sha256:
            expected = expected_sha256.lower()
            if resolved_path.is_dir():
                cand = _find_lora_weights_file(resolved_path)
                if cand is None:
                    raise FileNotFoundError(
                        "expected_sha256 provided, but no lora_weights.safetensors/.ckpt found to verify"
                    )
                actual = _sha256_file(cand)
            else:
                actual = _sha256_file(resolved_path)

            if actual.lower() != expected:
                raise RuntimeError(f"LoRA sha256 mismatch: expected={expected} actual={actual}")

        done_marker.parent.mkdir(parents=True, exist_ok=True)
        rel_path = resolved_path.resolve().relative_to(dst_root.resolve())
        resolved_path_marker.write_text(str(rel_path), encoding="utf-8")
        done_marker.write_text("ok", encoding="utf-8")
        return ResolvedArtifact(local_path=resolved_path, cache_key=key)
    finally:
        if tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)
