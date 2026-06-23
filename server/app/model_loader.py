"""Loads the VibeThinker causal LM once and exposes it as a process-wide singleton.

The model must load exactly once, at FastAPI startup, and stay resident for the
life of the process. Every request handler reads from this module instead of
loading its own copy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from app.config import settings

logger = logging.getLogger("vibethinker.model_loader")


class ModelLoadError(RuntimeError):
    """Raised when the model or tokenizer fails to load for any reason."""


@dataclass
class ModelState:
    """Holds the loaded model/tokenizer plus metadata used by /health and GraphQL."""

    model: Optional[PreTrainedModel] = None
    tokenizer: Optional[PreTrainedTokenizerBase] = None
    device: str = "cpu"
    model_source: str = ""
    loaded_in_4bit: bool = False
    load_error: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None


state = ModelState()


def _resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_quantization_config():
    """Builds a 4-bit BitsAndBytesConfig, raising clearly if bitsandbytes is unavailable."""
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise ModelLoadError(
            "LOAD_IN_4BIT is enabled but bitsandbytes/transformers support for it "
            "is not installed. Install bitsandbytes or set LOAD_IN_4BIT=false."
        ) from exc

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def load_model() -> ModelState:
    """Loads tokenizer + model into the module-level `state` singleton.

    Safe to call once at startup. On failure, `state.load_error` is set and
    `state.is_loaded` stays False so request handlers can fail fast with a
    clear error instead of pretending a model is available.
    """
    model_source = settings.model_source
    device = _resolve_device()
    logger.info("Loading model '%s' on device '%s'", model_source, device)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)

        load_kwargs: dict = {"trust_remote_code": True}

        if settings.load_in_4bit:
            if device != "cuda":
                raise ModelLoadError(
                    "LOAD_IN_4BIT=true requires a CUDA device, but none was detected."
                )
            load_kwargs["quantization_config"] = _build_quantization_config()
            load_kwargs["device_map"] = "auto"
        elif device == "cuda":
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.float32

        model = AutoModelForCausalLM.from_pretrained(model_source, **load_kwargs)
        model.eval()

        if not settings.load_in_4bit and device != "cuda":
            model = model.to(device)

        state.model = model
        state.tokenizer = tokenizer
        state.device = device
        state.model_source = model_source
        state.loaded_in_4bit = settings.load_in_4bit
        state.load_error = None
        logger.info("Model loaded successfully on %s (4bit=%s)", device, settings.load_in_4bit)

    except Exception as exc:  # noqa: BLE001 - we deliberately capture and surface any load failure
        logger.exception("Model load failed")
        state.model = None
        state.tokenizer = None
        state.load_error = str(exc)

    return state
