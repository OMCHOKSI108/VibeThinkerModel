"""Lightweight keyword-pattern intent classification used to pick a response
formatting hint (see context_manager.py). No model call is involved — this is
plain substring matching against the raw user message.
"""

from __future__ import annotations

ONESHOT_COMMAND = "oneshot_command"
CODING_AGENT_PROMPT = "coding_agent_prompt"
DEBUG = "debug"
EXPLANATION = "explanation"
CODE_GENERATION = "code_generation"
CODE_REVIEW = "code_review"
REPO_ANALYSIS = "repo_analysis"
GENERAL_CHAT = "general_chat"


def detect_intent(message: str) -> str:
    text = message.lower()

    if "oneshot" in text or "one shot" in text or "one-shot" in text:
        return ONESHOT_COMMAND

    if "give me prompt" in text or "coding agent" in text or "coding-agent" in text:
        return CODING_AGENT_PROMPT

    if "error" in text or "traceback" in text or "exception" in text or "not working" in text:
        return DEBUG

    if "explain" in text or "what is" in text or "what's" in text or "meaning" in text:
        return EXPLANATION

    if "create" in text or "build" in text or "implement" in text:
        return CODE_GENERATION

    if "review" in text or "audit" in text or "check this" in text:
        return CODE_REVIEW

    if "read this repo" in text or "github.com" in text:
        return REPO_ANALYSIS

    return GENERAL_CHAT
