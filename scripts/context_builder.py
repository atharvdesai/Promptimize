"""Deterministically pack prompt, recent chat, and retrieved memory for a gate.

This module has no model, database, retrieval, or network dependency. The
caller supplies already-retrieved context and a tokenizer callback for the
decision model. The default 4,096-token budget is a conservative working
budget for a model with an 8,192-token input limit, not the model's hard limit.

Input shape::

    {
      "prompt": "...",
      "recent_context": [{"role": "user", "content": "..."}, ...],
      "memories": [{
        "id": "...", "content": "...", "source": "...",
        "status": "user_confirmed" | "tool_observed" | "assistant_unconfirmed",
        "relevance": 0.0,
        "updated_at": "..."
      }, ...]
    }

Recent messages are supplied oldest-to-newest. Section targets are soft
fractions of the capacity left after the mandatory prompt and packet framing.
Unused capacity is filled according to ``spillover_order``. Context records
are kept whole; the current prompt is never truncated.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

TOKEN_BUDGET = 4096
DEFAULT_SECTION_TARGETS = {"recent_context": 0.50, "memories": 0.30}
DEFAULT_SPILLOVER_ORDER = ("recent_context", "memories")
MEMORY_STATUSES = {"user_confirmed", "tool_observed", "assistant_unconfirmed"}
MESSAGE_ROLES = {"system", "developer", "user", "assistant", "tool"}
SECTIONS = ("recent_context", "memories")


class ContextBuildError(ValueError):
    """Raised when input or packing policy is invalid."""


class PromptExceedsBudget(ContextBuildError):
    """Raised when the unmodified prompt cannot fit in the supplied budget."""


def _serialize(packet: Mapping[str, Any]) -> str:
    return json.dumps(packet, ensure_ascii=False, separators=(",", ":"))


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContextBuildError(f"{field} must be a non-empty string")
    return value


def _validate_message(value: Any, index: int) -> dict[str, str]:
    field = f"recent_context[{index}]"
    if not isinstance(value, Mapping):
        raise ContextBuildError(f"{field} must be an object")
    role = value.get("role")
    if not isinstance(role, str) or role not in MESSAGE_ROLES:
        raise ContextBuildError(f"{field}.role must be one of {sorted(MESSAGE_ROLES)}")
    return {
        "role": role,
        "content": _require_text(value.get("content"), f"{field}.content"),
    }


def _validate_memory(value: Any, index: int) -> dict[str, Any]:
    field = f"memories[{index}]"
    if not isinstance(value, Mapping):
        raise ContextBuildError(f"{field} must be an object")
    status = value.get("status")
    if not isinstance(status, str) or status not in MEMORY_STATUSES:
        raise ContextBuildError(f"{field}.status must be one of {sorted(MEMORY_STATUSES)}")
    relevance = value.get("relevance")
    if isinstance(relevance, bool) or not isinstance(relevance, (int, float)):
        raise ContextBuildError(f"{field}.relevance must be a finite number")
    if not math.isfinite(relevance):
        raise ContextBuildError(f"{field}.relevance must be a finite number")

    item: dict[str, Any] = {
        "id": _require_text(value.get("id"), f"{field}.id"),
        "content": _require_text(value.get("content"), f"{field}.content"),
        "source": _require_text(value.get("source"), f"{field}.source"),
        "status": status,
        "relevance": float(relevance),
    }
    for optional in ("updated_at", "project_id"):
        if value.get(optional) is not None:
            item[optional] = _require_text(value[optional], f"{field}.{optional}")
    return item


def _validate_policy(
    section_targets: Mapping[str, float] | None,
    spillover_order: Sequence[str] | None,
) -> tuple[dict[str, float], tuple[str, ...]]:
    if section_targets is not None and not isinstance(section_targets, Mapping):
        raise ContextBuildError("section_targets must be an object")
    targets = dict(DEFAULT_SECTION_TARGETS if section_targets is None else section_targets)
    if set(targets) != set(SECTIONS):
        raise ContextBuildError(f"section_targets must contain exactly {SECTIONS}")
    for section, share in targets.items():
        if isinstance(share, bool) or not isinstance(share, (int, float)):
            raise ContextBuildError(f"section target for {section} must be a number from 0 to 1")
        if not math.isfinite(share) or not 0 <= share <= 1:
            raise ContextBuildError(f"section target for {section} must be a finite number from 0 to 1")
    if sum(targets.values()) > 1:
        raise ContextBuildError("section target shares must sum to no more than 1")

    if spillover_order is not None and (
        not isinstance(spillover_order, Sequence) or isinstance(spillover_order, (str, bytes))
    ):
        raise ContextBuildError("spillover_order must be a sequence of section names")
    order = tuple(DEFAULT_SPILLOVER_ORDER if spillover_order is None else spillover_order)
    if len(order) != len(SECTIONS) or set(order) != set(SECTIONS):
        raise ContextBuildError(f"spillover_order must list each section exactly once: {SECTIONS}")
    return {name: float(value) for name, value in targets.items()}, order


def build_context(
    data: Mapping[str, Any],
    token_counter: Callable[[str], int],
    *,
    token_budget: int = TOKEN_BUDGET,
    section_targets: Mapping[str, float] | None = None,
    spillover_order: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a JSON packet within a configurable token budget.

    ``section_targets`` are soft shares of the context capacity remaining
    after the complete prompt and empty packet framing. The unallocated share
    is a shared reserve. Unused section targets and that reserve spill to
    sections in ``spillover_order``. The caller can change these defaults
    without changing the packing algorithm.

    ``token_counter`` receives the serialized packet and must return its token
    count for the eventual decision model. No approximate fallback is used.
    """
    if not isinstance(data, Mapping):
        raise ContextBuildError("input must be an object")
    if not callable(token_counter):
        raise ContextBuildError("token_counter must be callable")
    if isinstance(token_budget, bool) or not isinstance(token_budget, int) or token_budget < 1:
        raise ContextBuildError("token_budget must be a positive integer")
    targets, order = _validate_policy(section_targets, spillover_order)

    prompt = _require_text(data.get("prompt"), "prompt")
    raw_recent = data.get("recent_context", [])
    raw_memories = data.get("memories", [])
    if not isinstance(raw_recent, Sequence) or isinstance(raw_recent, (str, bytes)):
        raise ContextBuildError("recent_context must be a list of messages")
    if not isinstance(raw_memories, Sequence) or isinstance(raw_memories, (str, bytes)):
        raise ContextBuildError("memories must be a list of records")

    recent = [_validate_message(item, i) for i, item in enumerate(raw_recent)]
    memories = [_validate_memory(item, i) for i, item in enumerate(raw_memories)]
    excluded_unconfirmed = [
        item["id"] for item in memories if item["status"] == "assistant_unconfirmed"
    ]
    eligible_memories = [
        item for item in memories if item["status"] != "assistant_unconfirmed"
    ]
    # The retrieval score is primary; trust status and timestamp break ties.
    status_rank = {"user_confirmed": 1, "tool_observed": 0}
    eligible_memories.sort(
        key=lambda item: (
            item["relevance"],
            status_rank[item["status"]],
            item.get("updated_at", ""),
        ),
        reverse=True,
    )

    packet: dict[str, Any] = {
        "version": 1,
        "prompt": prompt,
        "recent_context": [],
        "memories": [],
    }

    def count(value: Mapping[str, Any]) -> int:
        result = token_counter(_serialize(value))
        if isinstance(result, bool) or not isinstance(result, int) or result < 0:
            raise ContextBuildError("token_counter must return a non-negative integer")
        return result

    base_tokens = count(packet)
    if base_tokens > token_budget:
        raise PromptExceedsBudget(
            f"prompt and packet framing require {base_tokens} tokens; budget is {token_budget}"
        )
    available = token_budget - base_tokens
    target_tokens = {
        section: int(available * share) for section, share in targets.items()
    }
    selected_recent: set[int] = set()
    selected_memories: set[int] = set()

    def section_cost(section: str, items: list[Any]) -> int:
        trial = {
            "version": 1,
            "prompt": prompt,
            "recent_context": items if section == "recent_context" else [],
            "memories": items if section == "memories" else [],
        }
        return max(0, count(trial) - base_tokens)

    def try_add(section: str, index: int, *, target_limit: int | None) -> bool:
        if section == "recent_context":
            if index in selected_recent:
                return False
            current = [recent[i] for i in sorted(selected_recent)]
            indices = selected_recent
            records = recent
        else:
            if index in selected_memories:
                return False
            current = [eligible_memories[i] for i in sorted(selected_memories)]
            indices = selected_memories
            records = eligible_memories

        proposed_indices = sorted((*indices, index))
        proposed = [records[i] for i in proposed_indices]
        if target_limit is not None and section_cost(section, proposed) > target_limit:
            return False

        packet[section] = proposed
        if count(packet) > token_budget:
            packet[section] = current
            return False
        indices.add(index)
        return True

    # Fill each section up to its soft target. Newer chat is considered first;
    # retrieved memories are considered by relevance, then trust and recency.
    for index in range(len(recent) - 1, -1, -1):
        try_add("recent_context", index, target_limit=target_tokens["recent_context"])
    for index in range(len(eligible_memories)):
        try_add("memories", index, target_limit=target_tokens["memories"])

    # Any unused target and the shared reserve can be used by either section.
    for section in order:
        if section == "recent_context":
            candidates = range(len(recent) - 1, -1, -1)
        else:
            candidates = range(len(eligible_memories))
        for index in candidates:
            try_add(section, index, target_limit=None)

    packet["recent_context"] = [recent[i] for i in sorted(selected_recent)]
    selected_memory_ids = {eligible_memories[i]["id"] for i in selected_memories}
    omitted_recent = [f"recent:{i}" for i in range(len(recent)) if i not in selected_recent]
    omitted_memories = [
        item["id"] for item in eligible_memories if item["id"] not in selected_memory_ids
    ]
    serialized = _serialize(packet)
    final_count = token_counter(serialized)
    if isinstance(final_count, bool) or not isinstance(final_count, int) or final_count < 0:
        raise ContextBuildError("token_counter must return a non-negative integer")
    if final_count > token_budget:
        raise ContextBuildError("internal packing error: result exceeds token budget")

    return {
        "packet": packet,
        "serialized": serialized,
        "token_count": final_count,
        "token_budget": token_budget,
        "omitted_recent": omitted_recent,
        "omitted_memories": omitted_memories,
        "excluded_unconfirmed_memories": excluded_unconfirmed,
    }
