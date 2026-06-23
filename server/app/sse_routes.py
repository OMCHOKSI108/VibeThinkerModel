"""HTTP SSE endpoint for chat streaming.

Token streaming is implemented here over plain SSE rather than through GraphQL,
since GraphQL subscriptions add transport complexity (websockets/multipart)
that buys nothing for a single linear token stream.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from app.config import settings
from app.context_manager import build_messages
from app.generation import stream_chat
from app.intent_detector import detect_intent
from app.memory_store import memory_store
from app.utils import StopSignal, format_sse_event

logger = logging.getLogger("vibethinker.sse_routes")

router = APIRouter()

_SENTINEL = object()


class ChatStreamRequest(BaseModel):
    """Validated body for POST /api/chat/stream. Pydantic rejects malformed
    requests with a 422 before any model work happens."""

    session_id: Optional[str] = None
    message: str = Field(..., min_length=1, max_length=8000)
    system_prompt: Optional[str] = None
    max_new_tokens: int = Field(default=settings.max_new_tokens, ge=1, le=4096)
    temperature: float = Field(default=settings.temperature, ge=0.0, le=2.0)
    top_p: float = Field(default=settings.top_p, ge=0.0, le=1.0)


def _safe_next(gen):
    """Wraps next() so StopIteration never has to cross an asyncio.to_thread
    boundary, where it would be misinterpreted as the future simply having no
    result (asyncio forbids raising StopIteration into a Future)."""
    try:
        return next(gen)
    except StopIteration:
        return _SENTINEL


@router.post("/api/chat/stream")
async def chat_stream(payload: ChatStreamRequest, request: Request) -> StreamingResponse:
    """Streams an SSE response: status events, then token events, then a
    final done/error event. The generation itself runs on a worker thread;
    this coroutine just relays events and watches for client disconnects.

    Conversation context (prior turns + folded summary) is pulled from
    memory_store and assembled by context_manager before generation starts;
    the new turn is written back to memory_store only after a clean `done`."""
    session_id = payload.session_id or str(uuid.uuid4())
    intent = detect_intent(payload.message)
    messages = build_messages(
        session_id=session_id,
        user_message=payload.message,
        intent=intent,
        system_prompt_override=payload.system_prompt,
    )
    stop_signal = StopSignal()

    async def event_source():
        gen = stream_chat(
            messages=messages,
            max_new_tokens=payload.max_new_tokens,
            temperature=payload.temperature,
            top_p=payload.top_p,
            stop_signal=stop_signal,
        )
        assistant_text = ""
        completed_cleanly = False
        try:
            while True:
                if await request.is_disconnected():
                    stop_signal.set()

                item = await asyncio.to_thread(_safe_next, gen)
                if item is _SENTINEL:
                    break

                event_name, data = item
                if event_name == "token":
                    assistant_text += data.get("text", "")
                elif event_name == "done":
                    completed_cleanly = True
                    data = {**data, "session_id": session_id}
                yield format_sse_event(event_name, data)
        finally:
            stop_signal.set()
            if completed_cleanly:
                memory_store.add_message(session_id, "user", payload.message)
                if assistant_text.strip():
                    memory_store.add_message(session_id, "assistant", assistant_text)

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
