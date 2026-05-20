"""
llm_qwen.py — CliMedV2
======================

Multi-backend LLM wrapper cho Qwen 2.5 (và tương tự).

Backend 1: Ollama (HTTP API, default, dễ setup nhất)
Backend 2: HuggingFace transformers (+ bitsandbytes 4-bit, optional)

Cách dùng nhanh:
    from llm_qwen import get_llm_client
    llm = get_llm_client()              # auto: Ollama nếu chạy, không thì HF
    text = llm.chat([{"role": "user", "content": "xin chào"}])

    # Streaming:
    for chunk in llm.chat_stream(messages):
        print(chunk, end="", flush=True)

Cấu hình qua env (override default):
    CLIMED_LLM_BACKEND=ollama|hf|auto    # mặc định: auto
    CLIMED_LLM_MODEL=qwen2.5:7b          # Ollama tag, hoặc HF repo cho hf backend
    CLIMED_OLLAMA_HOST=http://localhost:11434
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

Message = Dict[str, str]   # {"role": "system|user|assistant", "content": "..."}

# Defaults — override qua env nếu cần
DEFAULT_BACKEND = os.getenv("CLIMED_LLM_BACKEND", "auto").lower()
DEFAULT_OLLAMA_MODEL = os.getenv("CLIMED_LLM_MODEL", "qwen2.5:7b")
DEFAULT_HF_MODEL = os.getenv("CLIMED_HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
DEFAULT_OLLAMA_HOST = os.getenv("CLIMED_OLLAMA_HOST", "http://localhost:11434")


# ===========================================================================
# Abstract base
# ===========================================================================

class LLMClient(ABC):
    """Giao diện chung. Mọi backend đều trả `chat` (sync) và `chat_stream`."""

    name: str = "abstract"

    @abstractmethod
    def is_available(self) -> bool:
        """Backend có sẵn sàng không (server đang chạy / lib đã cài)?"""
        ...

    @abstractmethod
    def chat(self, messages: List[Message],
             temperature: float = 0.3,
             max_tokens: int = 1024) -> str:
        """Gọi 1 lượt non-streaming, trả về full response string."""
        ...

    @abstractmethod
    def chat_stream(self, messages: List[Message],
                    temperature: float = 0.3,
                    max_tokens: int = 1024) -> Generator[str, None, None]:
        """Streaming: yield từng token/chunk."""
        ...


# ===========================================================================
# Backend 1: Ollama
# ===========================================================================

class OllamaClient(LLMClient):
    """
    Wrapper REST API của Ollama (http://localhost:11434).

    Ưu: setup 1 lệnh (`ollama pull qwen2.5:7b`), chạy được CPU.
    Nhược: phải có Ollama daemon chạy ngầm.
    """

    name = "ollama"

    def __init__(self,
                 model: str = DEFAULT_OLLAMA_MODEL,
                 host: str = DEFAULT_OLLAMA_HOST,
                 timeout: float = 180.0):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def is_available(self) -> bool:
        try:
            import requests
            r = requests.get(f"{self.host}/api/tags", timeout=2.0)
            if r.status_code != 200:
                return False
            # Check model đã pull chưa
            tags = [m["name"] for m in r.json().get("models", [])]
            if not any(self.model in t for t in tags):
                logger.warning("Ollama đang chạy nhưng chưa pull model '%s'. "
                               "Chạy: ollama pull %s",
                               self.model, self.model)
                return False
            return True
        except Exception as e:
            logger.debug("Ollama not available: %s", e)
            return False

    def chat(self, messages, temperature=0.3, max_tokens=1024):
        import requests
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": 0.9,
            },
        }
        r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def chat_stream(self, messages, temperature=0.3, max_tokens=1024):
        import requests
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": 0.9,
            },
        }
        with requests.post(f"{self.host}/api/chat", json=payload,
                           timeout=self.timeout, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = data.get("message", {})
                chunk = msg.get("content", "")
                if chunk:
                    yield chunk
                if data.get("done"):
                    break


# ===========================================================================
# Backend 2: HuggingFace transformers
# ===========================================================================

class HFClient(LLMClient):
    """
    Wrapper transformers + bitsandbytes 4-bit.

    Ưu: full control, dễ fine-tune, dùng được mọi HF checkpoint.
    Nhược: tốn RAM/VRAM, lần đầu load chậm (~15GB download cho 7B FP16).
    """

    name = "hf"

    def __init__(self,
                 model_name: str = DEFAULT_HF_MODEL,
                 quantize: bool = True,
                 device_map: str = "auto"):
        self.model_name = model_name
        self.quantize = quantize
        self.device_map = device_map
        self._model = None
        self._tokenizer = None

    def is_available(self) -> bool:
        try:
            import transformers  # noqa: F401
            import torch         # noqa: F401
            return True
        except ImportError:
            return False

    def _load(self) -> None:
        """Lazy-load model (chỉ load khi gọi chat lần đầu)."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers / torch chưa cài. Chạy:\n"
                "  pip install transformers accelerate torch"
            ) from e

        kwargs: Dict[str, Any] = {
            "device_map": self.device_map,
            "torch_dtype": torch.float16,
        }

        # Quantization 4-bit (giảm VRAM ~4×)
        if self.quantize:
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                # Khi dùng quantization thì không truyền torch_dtype nữa
                kwargs.pop("torch_dtype", None)
            except ImportError:
                logger.warning("bitsandbytes chưa cài — load full precision. "
                               "Cài: pip install bitsandbytes")

        logger.info("Loading %s ... (lần đầu sẽ download ~15GB)", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name,torch_dtype="auto",device_map="auto", **kwargs)
        logger.info("Loaded.")

    def _build_inputs(self, messages: List[Message]):
        """Apply Qwen chat template → tokenize."""
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        return self._tokenizer([text], return_tensors="pt").to(self._model.device)

    def chat(self, messages, temperature=0.3, max_tokens=1024):
        self._load()
        inputs = self._build_inputs(messages)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-6),
            top_p=0.9,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        gen_ids = outputs[0, inputs.input_ids.shape[1]:]
        return self._tokenizer.decode(gen_ids, skip_special_tokens=True)

    def chat_stream(self, messages, temperature=0.3, max_tokens=1024):
        from threading import Thread
        from transformers import TextIteratorStreamer

        self._load()
        inputs = self._build_inputs(messages)

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True,
        )
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-6),
            top_p=0.9,
            pad_token_id=self._tokenizer.eos_token_id,
            streamer=streamer,
        )
        thread = Thread(target=self._model.generate, kwargs=gen_kwargs)
        thread.start()
        for chunk in streamer:
            yield chunk


# ===========================================================================
# Factory: auto-pick backend
# ===========================================================================

def get_llm_client(backend: str = DEFAULT_BACKEND,
                   model: Optional[str] = None) -> LLMClient:
    """
    Trả về LLMClient phù hợp.

    backend = "auto"        → ưu tiên Ollama, fallback HF
    backend = "ollama"      → bắt buộc Ollama
    backend = "hf"          → bắt buộc HuggingFace
    """
    backend = backend.lower()

    if backend in ("ollama",):
        client = OllamaClient(model=model or DEFAULT_OLLAMA_MODEL)
        if not client.is_available():
            raise RuntimeError(
                f"Ollama không sẵn sàng. Kiểm tra:\n"
                f"  1. Ollama daemon chạy chưa? (ollama serve)\n"
                f"  2. Đã pull model chưa? (ollama pull {client.model})"
            )
        return client

    if backend in ("hf", "transformers", "huggingface"):
        client = HFClient(model_name=model or DEFAULT_HF_MODEL)
        if not client.is_available():
            raise RuntimeError(
                "transformers/torch chưa cài. "
                "Chạy: pip install transformers accelerate torch bitsandbytes"
            )
        return client

    if backend == "auto":
        # 1) Thử Ollama
        ollama = OllamaClient(model=model or DEFAULT_OLLAMA_MODEL)
        if ollama.is_available():
            logger.info("Sử dụng backend Ollama (%s)", ollama.model)
            return ollama
        # 2) Fallback HF
        hf = HFClient(model_name=model or DEFAULT_HF_MODEL)
        if hf.is_available():
            logger.info("Sử dụng backend HuggingFace (%s)", hf.model_name)
            return hf
        raise RuntimeError(
            "Không tìm thấy backend LLM nào.\n"
            "Lựa chọn:\n"
            "  A) Cài Ollama: brew install ollama && ollama pull qwen2.5:7b\n"
            "  B) Cài transformers: pip install transformers accelerate torch bitsandbytes"
        )

    raise ValueError(f"Backend không hợp lệ: {backend!r}")


# ===========================================================================
# CLI test
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    llm = get_llm_client()
    print(f"Backend: {llm.name}")
    print()
    print("--- Streaming response ---")
    msgs = [
        {"role": "system",
         "content": "Bạn là trợ lý y khoa nói tiếng Việt, ngắn gọn."},
        {"role": "user",
         "content": "Tiểu cầu 50 K/uL ở bệnh nhân sốt xuất huyết ngày 5 có nguy hiểm không?"},
    ]
    for chunk in llm.chat_stream(msgs):
        print(chunk, end="", flush=True)
    print()