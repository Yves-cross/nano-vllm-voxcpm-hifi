from __future__ import annotations

import asyncio
import queue
import threading
import time
from typing import AsyncIterator, Protocol

import numpy as np

from app.core.config import Mp3Config
from app.core.metrics import AUDIO_ENCODE_FAILURES_TOTAL, AUDIO_ENCODE_SECONDS


class _DisconnectableRequest(Protocol):
    async def is_disconnected(self) -> bool: ...


def float32_to_s16le_bytes(wav: np.ndarray) -> bytes:
    wav_f32 = wav.astype(np.float32, copy=False)
    wav_f32 = np.clip(wav_f32, -1.0, 1.0)
    wav_i16 = (wav_f32 * 32767.0).astype(np.int16, copy=False)
    return wav_i16.tobytes(order="C")


def encode_mp3_bytes(wav: np.ndarray, sample_rate: int, mp3: Mp3Config) -> bytes:
    """Encode a full float32 mono waveform into a single MP3 byte payload."""
    try:
        import lameenc

        enc = lameenc.Encoder()
        enc.set_bit_rate(mp3.bitrate_kbps)
        enc.set_in_sample_rate(sample_rate)
        enc.set_channels(1)
        enc.set_quality(mp3.quality)

        pcm_bytes = float32_to_s16le_bytes(wav)
        t0 = time.perf_counter()
        out = enc.encode(pcm_bytes)
        out += enc.flush()
        AUDIO_ENCODE_SECONDS.observe(time.perf_counter() - t0)
        if isinstance(out, (bytearray, memoryview)):
            out = bytes(out)
        return out or b""
    except BaseException:
        AUDIO_ENCODE_FAILURES_TOTAL.inc()
        raise


async def stream_mp3(
    *,
    request: _DisconnectableRequest,
    wav_chunks: AsyncIterator[np.ndarray],
    sample_rate: int,
    mp3: Mp3Config,
) -> AsyncIterator[bytes]:
    """Encode float32 mono waveform chunks to MP3 and stream bytes.

    Encoding is done in a background thread to avoid blocking the event loop.
    """

    pcm_q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=8)
    mp3_q: queue.Queue[bytes | None] = queue.Queue(maxsize=8)
    stop_evt = threading.Event()
    thread_exc: list[BaseException] = []

    def encoder_thread() -> None:
        try:
            import lameenc

            enc = lameenc.Encoder()
            enc.set_bit_rate(mp3.bitrate_kbps)
            enc.set_in_sample_rate(sample_rate)
            enc.set_channels(1)
            enc.set_quality(mp3.quality)

            encoded_any = False

            while True:
                item = pcm_q.get()
                if item is None or stop_evt.is_set():
                    break

                pcm_bytes = float32_to_s16le_bytes(item)
                t0 = time.perf_counter()
                out = enc.encode(pcm_bytes)
                encoded_any = True
                AUDIO_ENCODE_SECONDS.observe(time.perf_counter() - t0)

                # lameenc may return bytearray; StreamingResponse requires bytes.
                if out:
                    if isinstance(out, (bytearray, memoryview)):
                        out = bytes(out)
                    mp3_q.put(out)

            # Some lameenc builds raise if flush is called without encoding any samples.
            if encoded_any:
                out = enc.flush()
                if out:
                    if isinstance(out, (bytearray, memoryview)):
                        out = bytes(out)
                    mp3_q.put(out)
            mp3_q.put(None)
        except BaseException as e:
            AUDIO_ENCODE_FAILURES_TOTAL.inc()
            thread_exc.append(e)
            stop_evt.set()
            try:
                mp3_q.put(None)
            except Exception:
                pass

    enc_thread = threading.Thread(target=encoder_thread, name="mp3-encoder", daemon=True)
    enc_thread.start()

    async def pcm_producer() -> None:
        try:
            async for chunk in wav_chunks:
                if await request.is_disconnected():
                    stop_evt.set()
                    break
                await asyncio.to_thread(pcm_q.put, chunk)
        finally:
            try:
                await asyncio.to_thread(pcm_q.put, None)
            except Exception:
                pass

    producer_task = asyncio.create_task(pcm_producer())

    try:
        while True:
            item = await asyncio.to_thread(mp3_q.get)
            if item is None:
                break
            yield item
        if thread_exc:
            raise RuntimeError(f"MP3 encoder failed: {thread_exc[0]}")
    finally:
        stop_evt.set()
        producer_task.cancel()
        try:
            await producer_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        try:
            await asyncio.to_thread(pcm_q.put, None)
        except Exception:
            pass
        await asyncio.to_thread(enc_thread.join, 2.0)
