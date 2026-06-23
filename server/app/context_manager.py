"""Assembles the full role-tagged message list sent to the model for one turn.

Layers, in order:
  1. System prompt (identity/honesty/safety) + developer prompt (project
     rules) + a short intent-specific formatting hint.
  2. Folded summary of older turns, if any (see memory_store.py).
  3. Recent raw turns within the memory window.
  4. The current user message.

There is no retrieval/RAG layer — this app has no external context source to
pull from, and everything here runs against the locally loaded model only.
"""

from __future__ import annotations

from pathlib import Path

from app.intent_detector import (
    CODE_GENERATION,
    CODE_REVIEW,
    CODING_AGENT_PROMPT,
    DEBUG,
    EXPLANATION,
    ONESHOT_COMMAND,
    REPO_ANALYSIS,
)
from app.memory_store import memory_store

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

INTENT_HINTS = {
    ONESHOT_COMMAND: (
        "The user wants one ready-to-run command. Reply with a single fenced "
        "bash code block and minimal explanation."
    ),
    CODING_AGENT_PROMPT: (
        "The user wants a complete copy-paste prompt for a coding agent. Give "
        "one block with project goal, folder structure, implementation "
        "requirements, and verification steps. Tell the agent to make "
        "reasonable assumptions instead of asking questions."
    ),
    DEBUG: (
        "The user hit an error. Structure the reply as: what the error means, "
        "the most likely cause, the exact fix, updated code, and a command to "
        "verify it."
    ),
    EXPLANATION: "Give a short, direct explanation with a minimal example.",
    CODE_GENERATION: (
        "Provide complete, runnable code, name the files it goes in, and give "
        "the steps to run and verify it."
    ),
    CODE_REVIEW: "List findings with severity and a concrete fix for each.",
    REPO_ANALYSIS: (
        "Only describe what was actually provided in this conversation; do "
        "not invent file contents. State clearly if the repo content wasn't given."
    ),
}


def _read_prompt_file(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


_SYSTEM_PROMPT = _read_prompt_file("system_prompt.txt")
_DEVELOPER_PROMPT = _read_prompt_file("developer_prompt.txt")


def build_messages(
    session_id: str,
    user_message: str,
    intent: str,
    system_prompt_override: str | None = None,
) -> list[dict]:
    """Builds the ordered list of {role, content} messages for one model turn."""
    persona = (system_prompt_override or "").strip() or _SYSTEM_PROMPT

    system_sections = [persona, _DEVELOPER_PROMPT]
    hint = INTENT_HINTS.get(intent)
    if hint:
        system_sections.append(hint)

    messages: list[dict] = [{"role": "system", "content": "\n\n".join(system_sections)}]

    summary = memory_store.get_summary(session_id)
    if summary:
        messages.append(
            {
                "role": "system",
                "content": f"Summary of earlier turns in this conversation:\n{summary}",
            }
        )

    messages.extend(memory_store.get_recent_messages(session_id))
    messages.append({"role": "user", "content": user_message})
    return messages
