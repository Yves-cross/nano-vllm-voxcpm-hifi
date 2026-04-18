import base64
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import gradio as gr
import requests
import torch
from funasr import AutoModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DEFAULT_TARGET_TEXT = "今天天气不错，我们继续测试一下后续续写的稳定性。"
API_BASE_URL = os.environ.get("NANOVLLM_API_BASE_URL", "http://127.0.0.1:8800")
DEFAULT_CFG = float(os.environ.get("NANOVLLM_HIFI_DEFAULT_CFG", "2.0"))
DEFAULT_MAX_GENERATE_LENGTH = int(os.environ.get("NANOVLLM_HIFI_DEFAULT_MAX_GENERATE_LENGTH", "2000"))
DEFAULT_TEMPERATURE = float(os.environ.get("NANOVLLM_HIFI_DEFAULT_TEMPERATURE", "1.0"))

CUSTOM_CSS = """
.logo {text-align:center;margin: 0.5rem 0 1rem 0;}
.logo h1 {margin-bottom: 0.2rem;}
.logo p {color:#666; margin-top:0;}
"""


class NanoVllmHifiDemo:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Running ASR on device: {self.device}")
        self.asr_model = AutoModel(
            model="iic/SenseVoiceSmall",
            disable_update=True,
            log_level="DEBUG",
            device="cuda:0" if self.device == "cuda" else "cpu",
        )

    def recognize(self, prompt_wav: Optional[str]) -> str:
        if not prompt_wav:
            return ""
        res = self.asr_model.generate(input=prompt_wav, language="auto", use_itn=True)
        return res[0]["text"].split("|>")[-1]

    def generate_hifi(
        self,
        ref_wav: Optional[str],
        prompt_text: str,
        target_text: str,
        cfg_value: float,
        temperature: float,
        max_generate_length: int,
    ):
        if not ref_wav:
            raise gr.Error("请先上传参考音频")
        if not prompt_text.strip():
            raise gr.Error("请填写参考音频内容文本（prompt text）")
        if not target_text.strip():
            raise gr.Error("请填写目标文本")

        with open(ref_wav, "rb") as f:
            wav_b64 = base64.b64encode(f.read()).decode("ascii")

        add_payload = {
            "wav_base64": wav_b64,
            "wav_format": Path(ref_wav).suffix.lstrip(".") or "wav",
            "prompt_text": prompt_text.strip(),
        }
        add = requests.post(API_BASE_URL + "/add_hifi", json=add_payload, timeout=120)
        add.raise_for_status()
        add_obj = add.json()
        hifi_id = add_obj["hifi_id"]

        try:
            gen_payload = {
                "target_text": target_text.strip(),
                "hifi_id": hifi_id,
                "cfg_value": float(cfg_value),
                "temperature": float(temperature),
                "max_generate_length": int(max_generate_length),
            }
            r = requests.post(API_BASE_URL + "/generate_blocking_wav", json=gen_payload, timeout=120)
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(prefix="nanovllm_hifi_", suffix=".wav", delete=False) as tmp:
                tmp.write(r.content)
                out_path = tmp.name
            meta = (
                f"hifi_id={hifi_id}\n"
                f"cfg_value={cfg_value}\n"
                f"temperature={temperature}\n"
                f"max_generate_length={max_generate_length}\n"
                f"bytes={len(r.content)}"
            )
            return out_path, meta
        finally:
            try:
                requests.delete(API_BASE_URL + f"/hifi/{hifi_id}", timeout=30)
            except Exception as e:
                logger.warning(f"cleanup hifi_id failed: {e}")


def create_demo() -> gr.Blocks:
    demo = NanoVllmHifiDemo()
    with gr.Blocks(css=CUSTOM_CSS) as ui:
        gr.HTML(
            '<div class="logo"><h1>Nano-vLLM HiFi 克隆</h1><p>前端页面，后端直连 Nano-vLLM FastAPI 的 hifi API</p></div>'
        )
        gr.Markdown("上传参考音频后，可先点 **ASR 自动填充**，再手动修正 prompt text，最后生成 HiFi 克隆结果。")
        with gr.Row():
            with gr.Column():
                ref_wav = gr.Audio(sources=["upload", "microphone"], type="filepath", label="参考音频")
                asr_btn = gr.Button("ASR 自动填充", variant="secondary")
                prompt_text = gr.Textbox(label="参考音频内容文本（prompt text）", lines=4)
                target_text = gr.Textbox(label="目标文本", lines=3, value=DEFAULT_TARGET_TEXT)
                with gr.Accordion("高级参数", open=False):
                    cfg_value = gr.Slider(1.0, 3.0, value=DEFAULT_CFG, step=0.1, label="cfg_value")
                    temperature = gr.Slider(
                        0.0,
                        1.5,
                        value=DEFAULT_TEMPERATURE,
                        step=0.1,
                        label="temperature",
                    )
                    max_generate_length = gr.Slider(
                        64,
                        4096,
                        value=DEFAULT_MAX_GENERATE_LENGTH,
                        step=1,
                        label="max_generate_length",
                    )
                run_btn = gr.Button("开始生成", variant="primary", size="lg")
            with gr.Column():
                audio_out = gr.Audio(label="生成结果")
                meta_out = gr.Textbox(label="本次请求参数", lines=8)

        asr_btn.click(
            fn=demo.recognize,
            inputs=[ref_wav],
            outputs=[prompt_text],
            show_progress=True,
        )
        run_btn.click(
            fn=demo.generate_hifi,
            inputs=[
                ref_wav,
                prompt_text,
                target_text,
                cfg_value,
                temperature,
                max_generate_length,
            ],
            outputs=[audio_out, meta_out],
            show_progress=True,
            api_name="generate_hifi",
        )
    return ui


def run_demo(port: int = 8805):
    ui = create_demo()
    ui.queue(max_size=10, default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
        share=False,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8805)
    args = parser.parse_args()
    run_demo(port=args.port)
