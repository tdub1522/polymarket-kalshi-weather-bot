"""Planner — fast-tier decomposer that generates 3-5 atomic sub-questions.

Default plan (used when LLM is unavailable) mirrors the repo's standard
KXBTC15M decomposition: momentum, orderbook, P(BTC>strike), edge vs market.
"""
from __future__ import annotations

import json
from typing import Dict, List

from ..llm import Llm

_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "subtasks": {
            "type": "array",
            "minItems": 3,
            "maxItems": 5,
            "items": {"type": "string"},
        }
    },
    "required": ["subtasks"],
}

DEFAULT_SUBTASKS = [
    "What does BTC's 1h and 15m momentum signal about direction over the next "
    "few minutes given the time remaining in the market window?",
    "What does the Kalshi orderbook (yes_bid, yes_ask, volume, recent prints) "
    "reveal about market participants' lean on this contract?",
    "What is the model probability that BTC closes above the strike at expiry, "
    "given current spot, distance to strike, time decay, and realized volatility?",
    "Is there a meaningful edge between the model probability and the market-implied "
    "probability (yes_ask)? If so, in which direction and by how much?",
]


async def generate_subtasks(llm: Llm, goal: str, context: Dict) -> List[str]:
    """Decompose a trading goal into independent, parallelizable sub-questions."""
    ctx_summary = json.dumps(_compact_context(context), default=str)[:1800]
    prompt = (
        f"You are decomposing a Kalshi KXBTC15M trading question into 3-5 atomic, "
        f"parallelizable sub-questions. Each sub-question must be answerable "
        f"independently (no cross-references) and should target a distinct "
        f"analytical angle (momentum, orderbook, probability, edge, time decay).\n\n"
        f"Goal: {goal}\n\nContext: {ctx_summary}\n\n"
        f"Return JSON: {{\"subtasks\": [...]}}"
    )
    res = await llm.tool_call_json(prompt, schema=_PLAN_SCHEMA, tier="fast", max_tokens=600)
    if res.parsed_json and isinstance(res.parsed_json.get("subtasks"), list):
        subs = [s for s in res.parsed_json["subtasks"] if isinstance(s, str) and s.strip()]
        if 3 <= len(subs) <= 5:
            return subs
    return DEFAULT_SUBTASKS


def _compact_context(context: Dict) -> Dict:
    """Trim context for the prompt — keep only signal-relevant scalars."""
    keep = (
        "ticker", "strike", "yes_ask", "yes_bid", "no_ask", "no_bid",
        "volume", "spot", "spot_1h_change", "spot_15m_change",
        "seconds_to_expiry", "market_implied_p",
    )
    out: Dict = {}
    for k in keep:
        if k in context:
            out[k] = context[k]
    if "price_history" in context and isinstance(context["price_history"], list):
        # Last few points only.
        out["price_history_tail"] = context["price_history"][-6:]
    return out
