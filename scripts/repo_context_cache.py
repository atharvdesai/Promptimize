"""Cached, host-agent-summarized repo facts for the cascading context builder.

Generating a good compact summary of a file (keywords + one-line description)
is exactly the kind of thing an LLM is good at and a plain script is not --
but calling a model on every file on every prompt defeats the point of a
cheap gate. So this module does the boring, deterministic half (figuring out
which files are new or changed since last time, and caching the rest) and
leaves the smart half to a caller-supplied ``summarizer``: a callback with
the shape ``summarizer(path, content) -> {"keywords": [...], "description": "..."}``.
In production that callback is the host coding agent (e.g. Claude Code)
asked to summarize just the files that changed; in tests it's a stub.

No network access happens here, and nothing is summarized unless its content
hash has actually changed since the cache was last written.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

Summarizer = Callable[[str, str], dict[str, Any]]


class RepoContextCacheError(ValueError):
    """Raised when the cache file or a summarizer result is malformed."""


def _hash_content(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _validate_summary(path: str, summary: Any) -> dict[str, Any]:
    if not isinstance(summary, dict):
        raise RepoContextCacheError(f"summarizer result for {path!r} must be an object")
    keywords = summary.get("keywords")
    description = summary.get("description")
    if not isinstance(keywords, list) or not all(isinstance(k, str) for k in keywords):
        raise RepoContextCacheError(f"summarizer result for {path!r} must have a list[str] 'keywords'")
    if not isinstance(description, str) or not description.strip():
        raise RepoContextCacheError(f"summarizer result for {path!r} must have a non-empty 'description'")
    return {"keywords": keywords, "description": description}


class RepoContextCache:
    """Persists compact, host-agent-generated summaries for repo files.

    The cache file is a flat JSON object: ``{path: {hash, keywords,
    description, summarized_at}}``. A file is only re-summarized when its
    content hash no longer matches what's cached.
    """

    def __init__(self, cache_path: str | Path):
        self.cache_path = Path(cache_path)

    def _load(self) -> dict[str, Any]:
        if not self.cache_path.exists():
            return {}
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RepoContextCacheError(f"cache file {self.cache_path} is not valid JSON") from exc

    def _save(self, cache: dict[str, Any]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")

    def refresh(self, files: dict[str, str], summarizer: Summarizer) -> dict[str, Any]:
        """Bring the cache up to date for exactly the given ``{path: content}``.

        Only files whose content hash changed (or that are new) are passed to
        ``summarizer``. Files no longer present in ``files`` are dropped from
        the cache. Returns the refreshed cache, already persisted to disk.
        """
        cache = self._load()
        updated: dict[str, Any] = {}
        for path, content in files.items():
            digest = _hash_content(content)
            cached = cache.get(path)
            if cached is not None and cached.get("hash") == digest:
                updated[path] = cached
                continue
            summary = _validate_summary(path, summarizer(path, content))
            updated[path] = {
                "hash": digest,
                "keywords": summary["keywords"],
                "description": summary["description"],
                "summarized_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        self._save(updated)
        return updated

    def to_repo_context(
        self,
        cache: dict[str, Any] | None = None,
        *,
        relevance: Callable[[str, dict[str, Any]], float] | None = None,
    ) -> list[dict[str, Any]]:
        """Convert cache entries into the ``repo_context`` shape the cascading
        context builder expects: ``{id, content, relevance}``, one per file,
        content compacted to its keywords and description (not the raw file).
        """
        entries = self._load() if cache is None else cache
        score = relevance or (lambda _path, _entry: 0.5)
        return [
            {
                "id": path,
                "content": f"keywords: {', '.join(entry['keywords'])} | {entry['description']}",
                "relevance": float(score(path, entry)),
            }
            for path, entry in entries.items()
        ]


def read_files(paths: Iterable[str | Path]) -> dict[str, str]:
    """Convenience loader: ``{str(path): file_text}`` for files that exist and decode as UTF-8."""
    result: dict[str, str] = {}
    for path in paths:
        p = Path(path)
        try:
            result[str(path)] = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
    return result
