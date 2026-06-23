"""GraphQL schema: status/introspection queries plus the resetSession mutation.

Token streaming is intentionally NOT exposed here — see sse_routes.py for why.
This schema only covers health/model metadata and session bookkeeping, which
are simple request/response operations that fit GraphQL naturally.
"""

from __future__ import annotations

import strawberry

from app.memory_store import memory_store
from app.model_loader import state as model_state


@strawberry.type
class HealthStatus:
    status: str
    model_loaded: bool
    device: str


@strawberry.type
class ModelInfo:
    model_source: str
    device: str
    loaded_in_4bit: bool
    is_loaded: bool
    load_error: str | None


@strawberry.type
class ResetSessionResult:
    session_id: str
    existed: bool


@strawberry.type
class Query:
    @strawberry.field(description="Reports whether the model is loaded and which device it's on.")
    def health(self) -> HealthStatus:
        return HealthStatus(
            status="ok" if model_state.is_loaded else "model_unavailable",
            model_loaded=model_state.is_loaded,
            device=model_state.device,
        )

    @strawberry.field(description="Returns metadata about the currently loaded model.")
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            model_source=model_state.model_source,
            device=model_state.device,
            loaded_in_4bit=model_state.loaded_in_4bit,
            is_loaded=model_state.is_loaded,
            load_error=model_state.load_error,
        )


@strawberry.type
class Mutation:
    @strawberry.mutation(description="Clears stored conversation memory for a chat session id.")
    def reset_session(self, session_id: str) -> ResetSessionResult:
        existed = memory_store.reset(session_id)
        return ResetSessionResult(session_id=session_id, existed=existed)


schema = strawberry.Schema(query=Query, mutation=Mutation)
