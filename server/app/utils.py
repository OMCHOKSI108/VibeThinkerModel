"""Small stateless/stateful helpers shared by the SSE route and GraphQL schema."""

from __future__ import annotations

import json
import threading
from typing import Any


def format_sse_event(event: str, data: dict[str, Any]) -> str:
    """Formats a single Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def build_prompt(tokenizer, messages: list[dict]) -> str:
    """Builds the model input text from a full list of role-tagged messages.

    Uses the tokenizer's chat template when the model ships one, since that
    matches how VibeThinker was instruction-tuned. Falls back to a plain
    role-tagged format for tokenizers without a chat template.
    """
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    lines = [f"{m['role'].capitalize()}: {m['content']}" for m in messages]
    return "\n\n".join(lines) + "\n\nAssistant:"


class ThinkTagStripper:
    """Filters `<think>...</think>` spans out of a stream of text deltas.

    The model is allowed to reason inside <think> tags, but that chain-of-thought
    must never reach the client as message content. This class consumes raw
    streamed deltas and yields only the text that falls outside think blocks,
    while correctly handling tags that get split across chunk boundaries.
    """

    OPEN_TAG = "<think>"
    CLOSE_TAG = "</think>"

    def __init__(self) -> None:
        self._buffer = ""
        self._inside_think = False
        self._saw_think_block = False

    @property
    def is_thinking(self) -> bool:
        return self._inside_think

    @property
    def saw_think_block(self) -> bool:
        return self._saw_think_block

    def feed(self, chunk: str) -> str:
        """Feeds a raw text delta in, returns the visible (non-think) text.

        Some generations close a think block with `</think>` without ever
        emitting a literal `<think>` first. In that case the text before the
        stray close tag was still reasoning, not an answer — we can't recall
        chunks already streamed to the client, but we can at least strip the
        literal close tag itself so it never appears as visible text.
        """
        self._buffer += chunk
        visible_parts: list[str] = []

        while True:
            if not self._inside_think:
                idx = self._buffer.find(self.OPEN_TAG)
                if idx == -1:
                    safe_len = min(
                        self._safe_emit_length(self._buffer, self.OPEN_TAG),
                        self._safe_emit_length(self._buffer, self.CLOSE_TAG),
                    )
                    segment = self._buffer[:safe_len]
                    visible_parts.append(segment.replace(self.CLOSE_TAG, ""))
                    self._buffer = self._buffer[safe_len:]
                    break
                segment = self._buffer[:idx]
                visible_parts.append(segment.replace(self.CLOSE_TAG, ""))
                self._buffer = self._buffer[idx + len(self.OPEN_TAG) :]
                self._inside_think = True
                self._saw_think_block = True
            else:
                idx = self._buffer.find(self.CLOSE_TAG)
                if idx == -1:
                    safe_len = self._safe_emit_length(self._buffer, self.CLOSE_TAG)
                    self._buffer = self._buffer[safe_len:]
                    break
                self._buffer = self._buffer[idx + len(self.CLOSE_TAG) :]
                self._inside_think = False

        return "".join(visible_parts)

    @staticmethod
    def _safe_emit_length(buffer: str, tag: str) -> int:
        """Returns how much of `buffer` is safe to emit without risking splitting `tag`."""
        max_check = min(len(tag) - 1, len(buffer))
        for length in range(max_check, 0, -1):
            if buffer[-length:] == tag[:length]:
                return len(buffer) - length
        return len(buffer)


class StopSignal:
    """A simple cross-thread flag used to cooperatively halt generation.

    Wraps a threading.Event so it can be passed into a transformers
    StoppingCriteria and be set from the SSE route when the client disconnects
    or asks to stop.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def set(self) -> None:
        self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()
