from dataclasses import dataclass
import logging
import os

import numpy as np
import torch
from transformers import LlamaTokenizerFast

from nanovllm_voxcpm.config import Config
from nanovllm_voxcpm.engine.llm_engine import LLMEngineBase
from nanovllm_voxcpm.engine.sequence import Sequence
from nanovllm_voxcpm.models.voxcpm2.config import VoxCPM2Config
from nanovllm_voxcpm.models.voxcpm2.runner import RunnerTask, VoxCPM2Payload, VoxCPM2Runner
from nanovllm_voxcpm.models.voxcpm2.utils import mask_multichar_chinese_tokens

logger = logging.getLogger("uvicorn.error")
SEQ_TRACE_ENABLED = os.environ.get("NANOVLLM_DEEP_TRACE", "1").strip().lower() not in ("0", "false", "off", "no")


@dataclass
class VoxCPM2SeqPayload:
    feats: list[np.ndarray]
    text_tokens: list[int]
    feat_masks: list[bool]
    generated_waveforms: list[np.ndarray]
    temperature: float
    cfg_value: float
    decode_pad: np.ndarray | None = None
    max_generate_length: int | None = None


class VoxCPM2Engine(LLMEngineBase):
    def __init__(self, config: Config[VoxCPM2Config]):
        self.n_decode_pad_frames = 12
        self.feat_dim = config.model_config.feat_dim
        self.patch_size = config.model_config.patch_size
        self.audio_start_token = 101
        self.ref_audio_start_token = 103
        self.ref_audio_end_token = 104

        self.block_size = config.kvcache_block_size
        self.max_model_len = config.max_model_len
        self.tokenizer = mask_multichar_chinese_tokens(LlamaTokenizerFast.from_pretrained(config.model))
        super().__init__(VoxCPM2Runner, config, config.tensor_parallel_size)

    def preprocess_seq(self, seq: Sequence[VoxCPM2SeqPayload], is_prefill: bool) -> RunnerTask[VoxCPM2Payload]:
        if is_prefill:
            if len(seq.custom_payload.feats) > 1:
                feats = np.concatenate(seq.custom_payload.feats, axis=0)
                seq.custom_payload.feats = [feats]

            payload = VoxCPM2Payload(
                text_tokens=np.array(seq.custom_payload.text_tokens[seq.num_cached_tokens :], dtype=np.int64),
                feats=seq.custom_payload.feats[-1][seq.num_cached_tokens :],
                feat_masks=np.array(seq.custom_payload.feat_masks[seq.num_cached_tokens :], dtype=np.bool_),
                temperature=seq.custom_payload.temperature,
                cfg_value=seq.custom_payload.cfg_value,
                padding_decode=seq.custom_payload.decode_pad,
            )
            if SEQ_TRACE_ENABLED:
                msg = (
                    f"voxcpm2_preprocess_trace is_prefill=1 seq_id={seq.seq_id} seq_len={len(seq)} cached={seq.num_cached_tokens} "
                    f"payload_tokens={payload.text_tokens.shape[0]} payload_feat_rows={payload.feats.shape[0]} "
                    f"payload_feat_true={int(payload.feat_masks.sum())}"
                )
                logger.info(msg)
                print(msg, flush=True)
            return RunnerTask(
                seq.block_table,
                len(seq),
                seq.num_cached_tokens,
                seq.block_size,
                payload,
            )

        payload = VoxCPM2Payload(
            text_tokens=np.array(seq.custom_payload.text_tokens[-1:], dtype=np.int64),
            feats=seq.custom_payload.feats[-1][-1:],
            feat_masks=np.array(seq.custom_payload.feat_masks[-1:], dtype=np.bool_),
            temperature=seq.custom_payload.temperature,
            cfg_value=seq.custom_payload.cfg_value,
            padding_decode=seq.custom_payload.decode_pad,
        )
        if SEQ_TRACE_ENABLED:
            msg = (
                f"voxcpm2_preprocess_trace is_prefill=0 seq_id={seq.seq_id} seq_len={len(seq)} cached={len(seq)-1} "
                f"payload_tokens={payload.text_tokens.shape[0]} payload_feat_rows={payload.feats.shape[0]} "
                f"payload_feat_true={int(payload.feat_masks.sum())}"
            )
            logger.info(msg)
            print(msg, flush=True)
        return RunnerTask(
            seq.block_table,
            len(seq),
            len(seq) - 1,
            seq.block_size,
            payload,
        )

    def postprocess_seq(self, seq: Sequence[VoxCPM2SeqPayload], outputs: dict, is_prefill: bool):
        stop_flag = outputs["stop_flag"]
        latents = outputs["latents"]
        waveforms = outputs["waveforms"]

        seq.append_token(latents.tobytes())
        seq.custom_payload.feats.append(latents[None])
        seq.custom_payload.text_tokens.append(0)
        seq.custom_payload.feat_masks.append(True)
        seq.custom_payload.generated_waveforms.append(waveforms)

        latents = latents.reshape(-1, self.feat_dim)
        if seq.custom_payload.decode_pad is not None:
            seq.custom_payload.decode_pad = np.concatenate([seq.custom_payload.decode_pad, latents], axis=0)[
                -self.n_decode_pad_frames :
            ]
        else:
            seq.custom_payload.decode_pad = latents[-self.n_decode_pad_frames :]

        if stop_flag == 1:
            seq.stoped = True
        elif (
            seq.custom_payload.max_generate_length is not None
            and len(seq.custom_payload.generated_waveforms) >= seq.custom_payload.max_generate_length
        ):
            seq.stoped = True

    def add_request(
        self,
        seq_id: str,
        target_text: str,
        prompt_text: str = "",
        prompt_latents: np.ndarray | None = None,
        ref_audio_latents: np.ndarray | None = None,
        max_generate_length: int = 2000,
        temperature: float = 1.0,
        cfg_value: float = 1.0,
    ):
        if max_generate_length < 1:
            raise ValueError(f"max_generate_length must be >= 1, got {max_generate_length}")

        text_tokens = self.tokenizer(prompt_text + target_text) + [self.audio_start_token]
        audio_feat = np.zeros((len(text_tokens), self.patch_size, self.feat_dim), dtype=np.float32)
        feat_masks = [False for _ in range(len(text_tokens))]
        hash_tokens = [t for t in text_tokens]
        decode_pad = None

        if ref_audio_latents is not None:
            wav_latents = ref_audio_latents
            wav_latents = wav_latents.reshape(-1, self.patch_size, self.feat_dim)

            audio_feat_pad = np.zeros((1, self.patch_size, self.feat_dim), dtype=np.float32)
            audio_feat = np.concatenate([audio_feat_pad, wav_latents, audio_feat_pad, audio_feat], axis=0)
            text_tokens = (
                [self.ref_audio_start_token]
                + ([0 for _ in range(wav_latents.shape[0])])
                + [self.ref_audio_end_token]
                + text_tokens
            )
            feat_masks = [False] + ([True for _ in range(wav_latents.shape[0])]) + [False] + feat_masks

            prepend_hash_tokens = (
                [self.ref_audio_start_token]
                + [wav_latents[i].tobytes() for i in range(wav_latents.shape[0])]
                + [self.ref_audio_end_token]
            )
            hash_tokens = prepend_hash_tokens + hash_tokens

        if prompt_latents is not None:
            wav_latents = prompt_latents
            decode_pad = wav_latents[-self.n_decode_pad_frames :]
            wav_latents = wav_latents.reshape(-1, self.patch_size, self.feat_dim)
            audio_feat = np.concatenate([audio_feat, wav_latents], axis=0)
            text_tokens.extend([0 for _ in range(wav_latents.shape[0])])
            feat_masks.extend([True for _ in range(wav_latents.shape[0])])
            for i in range(wav_latents.shape[0]):
                hash_tokens.append(wav_latents[i].tobytes())

        prompt_len = len(hash_tokens)
        total_len_upper_bound = prompt_len + max_generate_length
        if prompt_len > self.max_model_len:
            raise ValueError(
                f"Prompt is too long for max_model_len: prompt_len={prompt_len} > max_model_len={self.max_model_len}"
            )
        if total_len_upper_bound > self.max_model_len:
            raise ValueError(
                "Request may exceed max_model_len: "
                f"prompt_len({prompt_len}) + max_generate_length({max_generate_length}) = {total_len_upper_bound} "
                f"> max_model_len({self.max_model_len}). "
                "Reduce input length or max_generate_length, or increase max_model_len."
            )

        if SEQ_TRACE_ENABLED:
            text_token_len = len(text_tokens)
            feat_mask_true = int(sum(1 for x in feat_masks if x))
            prompt_latent_patches = 0 if prompt_latents is None else int(prompt_latents.shape[0] // self.patch_size)
            ref_latent_patches = 0 if ref_audio_latents is None else int(ref_audio_latents.shape[0] // self.patch_size)
            msg = (
                f"voxcpm2_add_request_trace seq_id={seq_id} target_text_len={len(target_text)} prompt_text_len={len(prompt_text)} "
                f"text_tokens={text_token_len} feat_true={feat_mask_true} prompt_patches={prompt_latent_patches} "
                f"ref_patches={ref_latent_patches} hash_tokens={len(hash_tokens)} max_generate_length={max_generate_length}"
            )
            logger.info(msg)
            print(msg, flush=True)

        seq = Sequence(
            seq_id,
            hash_tokens,
            self.block_size,
            VoxCPM2SeqPayload(
                feats=[audio_feat],
                text_tokens=text_tokens,
                feat_masks=feat_masks,
                decode_pad=decode_pad,
                temperature=temperature,
                cfg_value=cfg_value,
                max_generate_length=max_generate_length,
                generated_waveforms=[],
            ),
        )
        self.add_sequence(seq)

    def encode_latents(self, wav: torch.Tensor, align_size: int = -1) -> np.ndarray:
        if align_size == -1:
            align_size = self.patch_size * self.model_runner.vae.encoder_chunk_size
        if wav.size(1) % align_size != 0:
            remained = align_size - wav.size(1) % align_size
            wav = torch.nn.functional.pad(wav, (remained, 0))
        return self.model_runner.encode_latents(wav)
