"""Runs text generation against the loaded model and yields smart-buffered SSE events.

`stream_chat` is a synchronous generator: it starts `model.generate` on a
background thread (driven by a `TextIteratorStreamer`) and consumes raw text
deltas on the calling thread. The route layer is responsible for iterating
this generator off the asyncio event loop (see `sse_routes.py`).
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Iterator

import torch
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

from app.model_loader import state as model_state
from app.utils import StopSignal, ThinkTagStripper, build_prompt

logger = logging.getLogger("vibethinker.generation")

# Smart-streaming tuning: hold output back until either enough words have
# accumulated or enough time has passed, then flush in small word groups.
WARMUP_WORD_THRESHOLD = 10
WARMUP_TIME_SECONDS = 1.5
FLUSH_GROUP_WORDS = 3

_TOKEN_RE = re.compile(r"\S+\s*|\s+")


class GenerationError(RuntimeError):
    """Raised for request-time generation failures (model not loaded, etc.)."""


class _StopOnSignal(StoppingCriteria):
    """Lets `model.generate` be interrupted cooperatively from another thread."""

    def __init__(self, stop_signal: StopSignal) -> None:
        self._stop_signal = stop_signal

    def __call__(self, input_ids, scores, **kwargs) -> bool:  # noqa: ANN001
        return self._stop_signal.is_set()


def _split_complete_tokens(buffer: str) -> tuple[list[str], str]:
    """Splits buffered text into whitespace-bounded tokens, holding back a
    trailing token with no trailing whitespace since it may still be incomplete.
    """
    tokens = _TOKEN_RE.findall(buffer)
    if not tokens:
        return [], ""
    if tokens[-1] and not tokens[-1][-1].isspace():
        return tokens[:-1], tokens[-1]
    return tokens, ""


def _group_into_flushes(tokens: list[str], group_size: int) -> tuple[list[str], list[str]]:
    """Groups whitespace-bounded tokens into `group_size`-word chunks.

    Returns (ready_to_flush_chunks, leftover_tokens). Leftover tokens are the
    tail that did not complete a full group yet.
    """
    chunks: list[str] = []
    current: list[str] = []
    word_count = 0
    for tok in tokens:
        current.append(tok)
        if tok.strip():
            word_count += 1
        if word_count >= group_size:
            chunks.append("".join(current))
            current = []
            word_count = 0
    return chunks, current


def _is_out_of_memory(exc: Exception) -> bool:
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    return "out of memory" in str(exc).lower()


def _safe_error_message(exc: Exception) -> str:
    """Maps an internal exception to a message safe to show the client (no stack traces)."""
    if _is_out_of_memory(exc):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return (
            "GPU ran out of memory while generating. Try a shorter prompt, "
            "fewer max tokens, or enable LOAD_IN_4BIT."
        )
    return "Generation failed due to an internal error. Please try again."


def stream_chat(
    messages: list[dict],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stop_signal: StopSignal,
) -> Iterator[tuple[str, dict]]:
    """Yields (event_name, data) pairs: status/token/error/done.

    `messages` is the full role-tagged conversation context already assembled
    by context_manager.build_messages (system/developer/summary/history/current
    turn) — this function only turns it into model input and streams output.

    Buffers visible output until `WARMUP_WORD_THRESHOLD` words or
    `WARMUP_TIME_SECONDS` have elapsed, then flushes in `FLUSH_GROUP_WORDS`-word
    groups. Text inside `<think>...</think>` is never emitted as token content.
    """
    yield "status", {"stage": "queued", "message": "Queued..."}

    if not model_state.is_loaded:
        detail = model_state.load_error or "model has not finished loading"
        yield "error", {"message": f"Model is not available: {detail}"}
        return

    if not messages or not (messages[-1].get("content") or "").strip():
        yield "error", {"message": "Prompt must not be empty."}
        return

    tokenizer = model_state.tokenizer
    model = model_state.model

    try:
        prompt_text = build_prompt(tokenizer, messages)
        inputs = tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to build prompt/tokenize input")
        yield "error", {"message": _safe_error_message(exc)}
        return

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    generate_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        top_p=top_p,
        pad_token_id=pad_token_id,
        streamer=streamer,
        stopping_criteria=StoppingCriteriaList([_StopOnSignal(stop_signal)]),
    )

    worker_errors: list[Exception] = []

    def _worker() -> None:
        try:
            with torch.inference_mode():
                model.generate(**generate_kwargs)
        except Exception as exc:  # noqa: BLE001
            worker_errors.append(exc)

    thread = threading.Thread(target=_worker, daemon=True)

    yield "status", {"stage": "thinking", "message": "Thinking..."}

    thread.start()
    start_time = time.time()

    think_stripper = ThinkTagStripper()
    pending_buffer = ""
    started_streaming = False
    sent_streaming_status = False

    for raw_delta in streamer:
        if stop_signal.is_set():
            break

        visible_delta = think_stripper.feed(raw_delta)
        if not visible_delta:
            continue

        pending_buffer += visible_delta
        complete_tokens, partial = _split_complete_tokens(pending_buffer)

        if not started_streaming:
            word_count = sum(1 for tok in complete_tokens if tok.strip())
            elapsed = time.time() - start_time
            if word_count >= WARMUP_WORD_THRESHOLD or elapsed >= WARMUP_TIME_SECONDS:
                started_streaming = True
            else:
                pending_buffer = "".join(complete_tokens) + partial
                continue

        if not sent_streaming_status:
            yield "status", {"stage": "streaming", "message": "Streaming response..."}
            sent_streaming_status = True

        chunks, leftover_tokens = _group_into_flushes(complete_tokens, FLUSH_GROUP_WORDS)
        for chunk in chunks:
            yield "token", {"text": chunk}
        pending_buffer = "".join(leftover_tokens) + partial

    thread.join(timeout=30)

    if pending_buffer:
        yield "token", {"text": pending_buffer}

    if worker_errors:
        logger.error("Generation worker failed", exc_info=worker_errors[0])
        yield "error", {"message": _safe_error_message(worker_errors[0])}
        return

    finish_reason = "aborted" if stop_signal.is_set() else "stop"
    yield "done", {"finish_reason": finish_reason}
