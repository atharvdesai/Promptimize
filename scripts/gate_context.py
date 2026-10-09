"""Single entry point for building the gate's full decision context.

Combines the three inputs the decision model should see:

1. Recent chat, cascaded by recency (via ``build_cascading_context``).
2. A compact, cached, host-agent-summarized view of the repo (via
   ``RepoContextCache``) instead of raw file contents.
3. A cheap topic-continuity score (via ``score_topic_continuity``) flagging
   when this prompt looks unrelated to the last few turns -- a likely signal
   that something new and possibly ambiguous is being asked.

No model call happens inside this module itself except through the
caller-supplied ``summarizer`` (see repo_context_cache.py), which is where a
host coding agent plugs in to generate repo summaries -- and that only runs
for files whose content actually changed since the last call.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .continuity import score_topic_continuity
from .context_builder import TOKEN_BUDGET, build_cascading_context
from .repo_context_cache import RepoContextCache, Summarizer, read_files


def build_gate_context(
    prompt: str,
    token_counter: Callable[[str], int],
    *,
    recent_context: Sequence[Mapping[str, str]] = (),
    memories: Sequence[Mapping[str, Any]] = (),
    repo_files: Sequence[str | Path] = (),
    summarizer: Summarizer | None = None,
    cache_path: str | Path = ".promptimize/repo_context_cache.json",
    repo_relevance: Callable[[str, dict[str, Any]], float] | None = None,
    continuity_window: int = 3,
    token_budget: int = TOKEN_BUDGET,
    max_tiers: int = 10,
) -> dict[str, Any]:
    """Build the cascaded context packet plus a topic-continuity signal.

    ``repo_files`` are paths to summarize (typically a short, pre-selected
    list -- e.g. files the retrieval step thinks are relevant -- not the
    whole repo). Pass ``summarizer`` only when you have new/changed files to
    summarize; omit it to just reuse whatever is already cached. Omitting
    both ``repo_files`` and ``summarizer`` skips repo context entirely.
    """
    repo_context: list[dict[str, Any]] = []
    if repo_files:
        cache = RepoContextCache(cache_path)
        if summarizer is not None:
            files = read_files(repo_files)
            raw_cache = cache.refresh(files, summarizer)
        else:
            raw_cache = cache._load()  # reuse whatever's already cached, no re-summarizing
            raw_cache = {path: entry for path, entry in raw_cache.items() if path in {str(p) for p in repo_files}}
        repo_context = cache.to_repo_context(raw_cache, relevance=repo_relevance)

    cascade = build_cascading_context(
        {
            "prompt": prompt,
            "recent_context": list(recent_context),
            "memories": list(memories),
            "repo_context": repo_context,
        },
        token_counter,
        token_budget=token_budget,
        max_tiers=max_tiers,
    )

    continuity = score_topic_continuity(prompt, list(recent_context), window=continuity_window)

    return {**cascade, "topic_continuity": continuity}
