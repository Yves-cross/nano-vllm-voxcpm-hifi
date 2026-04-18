from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Standard health response for liveness/readiness endpoints."""

    status: Literal["ok"] = "ok"


class ErrorResponse(BaseModel):
    """Default error response shape produced by FastAPI for HTTPException.

    Note: FastAPI's validation errors (422) use a different schema.
    """

    detail: str = Field(..., description="Human-readable error message.")


class ModelInfo(BaseModel):
    """Read-only model metadata returned by the engine."""

    sample_rate: int = Field(..., description="Audio sample rate in Hz.", examples=[16000])
    channels: int = Field(..., description="Number of audio channels.", examples=[1])
    feat_dim: int = Field(..., description="Latent feature dimension.", examples=[64])
    patch_size: int = Field(..., description="Model patch size.", examples=[2])
    model_path: str = Field(
        ...,
        description="Resolved model path used by this instance.",
        examples=["/models/VoxCPM1.5"],
    )


class Mp3Info(BaseModel):
    """MP3 encoder configuration used by /generate."""

    bitrate_kbps: int | None = Field(None, description="Constant bitrate used for MP3 encoding.", examples=[192])
    quality: int | None = Field(None, description="LAME quality preset (0 is best, 2 is fast).", examples=[2])


class LoRAInfo(BaseModel):
    """LoRA startup configuration and load status."""

    lora_uri: str | None = Field(
        None,
        description="LoRA artifact URI (if configured).",
        examples=["hf://org/repo@main?path=step_0002000"],
    )
    lora_id: str | None = Field(
        None,
        description="Logical LoRA ID associated with this instance.",
        examples=["my-lora"],
    )
    cache_dir: str | None = Field(None, description="Cache directory used to resolve LoRA artifacts.")
    loaded: bool = Field(
        ...,
        description="Whether LoRA weights have been loaded and enabled.",
        examples=[False],
    )


class InfoResponse(BaseModel):
    """Response for GET /info."""

    model: ModelInfo
    lora: LoRAInfo
    mp3: Mp3Info


class EncodeLatentsRequest(BaseModel):
    """Request body for POST /encode_latents."""

    wav_base64: str = Field(
        ...,
        description="Base64-encoded audio file bytes (entire file contents). Do not include a data URI prefix.",
        examples=["UklGRiQAAABXQVZFZm10IBAAAAABAAEA..."],
    )
    wav_format: str = Field(
        ...,
        description="Audio container format for decoding (e.g. 'wav', 'flac', 'mp3'); passed to torchaudio.",
        examples=["wav"],
    )


class EncodeLatentsResponse(BaseModel):
    """Response body for POST /encode_latents."""

    prompt_latents_base64: str
    feat_dim: int
    latents_dtype: Literal["float32"] = "float32"
    sample_rate: int
    channels: int


class AddPromptRequest(BaseModel):
    """Request body for POST /add_prompt."""

    wav_base64: str = Field(..., description="Base64-encoded prompt audio file bytes.")
    wav_format: str = Field(..., description="Prompt audio container format (e.g. wav/flac/mp3).")
    prompt_text: str = Field(..., description="Transcript text corresponding to the prompt audio.")


class AddHiFiRequest(BaseModel):
    """Request body for POST /add_hifi."""

    wav_base64: str = Field(..., description="Base64-encoded reference/prompt audio file bytes.")
    wav_format: str = Field(..., description="Audio container format (e.g. wav/flac/mp3).")
    prompt_text: str = Field(..., description="Transcript text corresponding to the reference audio.")


class AddPromptResponse(BaseModel):
    """Response body for POST /add_prompt."""

    prompt_id: str
    feat_dim: int
    sample_rate: int
    channels: int


class AddReferenceRequest(BaseModel):
    """Request body for POST /add_reference."""

    wav_base64: str = Field(..., description="Base64-encoded reference audio file bytes.")
    wav_format: str = Field(..., description="Reference audio container format (e.g. wav/flac/mp3).")


class AddReferenceResponse(BaseModel):
    """Response body for POST /add_reference."""

    reference_id: str
    feat_dim: int
    sample_rate: int
    channels: int


class AddHiFiResponse(BaseModel):
    """Response body for POST /add_hifi."""

    hifi_id: str
    prompt_id: str
    reference_id: str
    feat_dim: int
    sample_rate: int
    channels: int


class GenerateRequest(BaseModel):
    """Request body for POST /generate.

    Prompt forms (mutually exclusive):

    - Zero-shot: omit all prompt_* fields.
    - WAV prompt: set prompt_wav_base64 + prompt_wav_format + prompt_text.
    - Latents prompt: set prompt_latents_base64 + prompt_text.
    - Cached prompt: set prompt_id.
    - Cached HiFi clone: set hifi_id.

    Reference audio (optional, mutually exclusive within the ref_audio_* group):

    - WAV reference: set ref_audio_wav_base64 + ref_audio_wav_format.
    - Latents reference: set ref_audio_latents_base64.
    - Cached reference: set ref_audio_id.
    """

    target_text: str = Field(..., description="Text to synthesize.")

    # Prompt forms (mutually exclusive):
    prompt_wav_base64: str | None = Field(
        None,
        description="(wav prompt) Base64-encoded audio file bytes (entire file contents).",
    )
    prompt_wav_format: str | None = Field(
        None,
        description="(wav prompt) Audio container format for decoding (e.g. 'wav', 'flac', 'mp3').",
    )
    prompt_latents_base64: str | None = Field(
        None,
        description="(latents prompt) Base64-encoded float32 bytes returned by /encode_latents.",
    )
    prompt_text: str | None = Field(
        None,
        description="Prompt transcript text. Required for wav/latents prompt; omitted for zero-shot or prompt_id.",
    )
    prompt_id: str | None = Field(
        None,
        description="(cached prompt) Stored prompt ID returned by /add_prompt.",
    )
    hifi_id: str | None = Field(
        None,
        description="(cached HiFi clone) Stored HiFi ID returned by /add_hifi.",
    )

    ref_audio_wav_base64: str | None = Field(
        None,
        description="(reference audio) Base64-encoded audio file bytes (entire file contents).",
    )
    ref_audio_wav_format: str | None = Field(
        None,
        description="(reference audio) Audio container format for decoding (e.g. 'wav', 'flac', 'mp3').",
    )
    ref_audio_latents_base64: str | None = Field(
        None,
        description="(reference audio) Base64-encoded float32 bytes returned by /encode_latents.",
    )
    ref_audio_id: str | None = Field(
        None,
        description="(reference audio) Cached reference ID returned by /add_reference.",
    )

    max_generate_length: int = Field(2000, ge=1, description="Maximum number of model generation steps.")
    temperature: float = Field(1.0, ge=0.0, description="Sampling temperature.")
    cfg_value: float = Field(2.0, ge=0.0, description="Classifier-free guidance scale.")
