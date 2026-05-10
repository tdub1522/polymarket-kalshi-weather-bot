"""Atomizer — fast binary gate: is this goal atomic or does it need decomposition?

For our use case (BTC 15m market analysis) the top-level goal is never atomic;
this exists for fidelity to the ROMA architecture and to allow recursion later.
"""
from __future__ import annotations

from typing import Dict
from ..llm import Llm

_ATOMIZE_SCHEMA = {
    "type": "object",
    "properties": {
        "atomic": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["atomic", "reason"],
}


async def is_atomic(llm: Llm, goal: str, context: Dict) -> bool:
    """Return True if the goal can be answered directly without decomposition."""
    # Heuristic short-circuit: top-level "Decide..." goals are never atomic.
    if goal.lower().startswith(("decide", "analyze", "produce a thesis")):
        return False

    prompt = (
        f"Goal: {goal}\n\n"
        f"Context keys: {list(context.keys())}\n\n"
        "Can this goal be answered with a single focused analysis, or does it need "
        "to be broken into multiple parallel sub-questions? Reply with JSON."
    )
    res = await llm.tool_call_json(prompt, schema=_ATOMIZE_SCHEMA, tier="fast", max_tokens=200)
    if res.parsed_json is None:
        return False
    return bool(res.parsed_json.get("atomic", False))
