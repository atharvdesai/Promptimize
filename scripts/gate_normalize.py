#!/usr/bin/env python3
"""Shared input normalization for the Promptimize gate.

Trial T5 showed the decision pipeline must see the same normalized text
at training time and at inference time: path-like tokens are stripped
and whitespace collapsed before embedding. This module is the SINGLE
implementation of that contract - experiments and the shipped plugin
both import it, and tests pin the behavior.

Rationale (see experiments/t5_result.json): specific file names almost
never carry discoverability semantics, but their presence correlated
with labels in the seed dataset, so the signal is removed from the
input rather than learned.
"""
import re

# A token is "path-like" when it contains a slash or ends in a common
# code/config/doc file extension. Anchored to the whole token.
PATHLIKE = re.compile(
    r"^\S*(?:/|\.(?:py|tsx?|jsx?|json|ya?ml|md|go|rb|php|pkl|sql|css|html|sh|txt|toml|cfg|xml|csv|lock))\S*$",
    re.I,
)


def normalize(prompt: str) -> str:
    """Strip path-like tokens and collapse whitespace.

    Applied identically at train and inference time (T5 contract).
    Discoverability semantics (what information is missing) is preserved;
    file names almost never carry it. Edge-case cost (a path that DID
    matter) is a measured limitation, tracked in trials P2/P3.
    """
    toks = [t for t in prompt.split() if not PATHLIKE.match(t)]
    return re.sub(r"\s+", " ", " ".join(toks)).strip()


# Measured blind spot: long requests are the cell the current pipeline
# gets wrong (probe families 1+2: long-ambiguous recall 0/8). The gate
# logs this flag so dogfooding data separates blind-spot decisions from
# the rest. 200 chars is a HEURISTIC flag derived from the probe
# failure zone (every long-ambiguous probe exceeds it) - it is not a
# validated boundary; short prompts can be ambiguous and long ones clear.
BLIND_SPOT_MIN_CHARS = 200


def in_measured_blind_spot(prompt: str) -> bool:
    return len(prompt) >= BLIND_SPOT_MIN_CHARS
