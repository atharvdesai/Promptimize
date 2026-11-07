"""Cheap, local signal for whether a prompt continues the recent conversation.

This has no model/network dependency on purpose: it exists to catch the case
the user described -- "if someone is talking about something, then their next
prompt has nothing to do with the last couple of prompts, it's probably
something to look into." That is a *discontinuity* signal, not a context
budget decision, so it stands apart from scripts/context_builder.py and is
meant to be handed to the decision model as extra metadata alongside the
packed context, not used to decide what to pack.

The score is deliberately a cheap lexical overlap (Jaccard over non-stopword
tokens), not an embedding or model call -- the gate has to stay cheap.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "i", "if", "in", "into", "is", "it", "its", "of", "on",
    "or", "that", "the", "this", "to", "was", "were", "will", "with", "you",
    "your", "we", "do", "does", "did", "can", "could", "should", "would",
    "my", "me", "please", "also", "just", "so", "then", "there",
}

_WORD_RE = re.compile(r"[a-z0-9_]+")


def _tokenize(text: str) -> set[str]:
    words = _WORD_RE.findall(text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def score_topic_continuity(
    prompt: str,
    recent_context: Sequence[Any],
    *,
    window: int = 3,
) -> float | None:
    """Jaccard overlap between ``prompt`` and the last ``window`` messages.

    Returns ``None`` when there is no recent conversation to compare against
    (nothing to be discontinuous *from*). Returns a float in ``[0, 1]``
    otherwise -- 0 means no shared non-trivial vocabulary with recent chat
    (a likely topic jump, worth a closer ambiguity check), 1 means heavy
    overlap. Messages are plain ``{"role", "content"}`` dicts, same shape as
    ``context_builder``'s ``recent_context``; only ``content`` is used.
    """
    if window < 1:
        raise ValueError("window must be a positive integer")

    relevant = [
        m for m in recent_context[-window:]
        if isinstance(m, dict) and isinstance(m.get("content"), str) and m["content"].strip()
    ]
    if not relevant:
        return None

    prompt_tokens = _tokenize(prompt)
    if not prompt_tokens:
        # No meaningful vocabulary in the prompt itself -- nothing to judge.
        return None

    recent_tokens: set[str] = set()
    for message in relevant:
        recent_tokens |= _tokenize(message["content"])
    if not recent_tokens:
        return None

    intersection = prompt_tokens & recent_tokens
    union = prompt_tokens | recent_tokens
    return len(intersection) / len(union)
