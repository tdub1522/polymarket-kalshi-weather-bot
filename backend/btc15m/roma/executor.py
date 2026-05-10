"""Executor — fast-tier worker that answers a single atomic sub-question."""
from __future__ import annotations

import json
from typing import Dict
from ..llm import Llm


async def execute_subtask(llm: Llm, subtask: str, context: Dict) -> str:
    """Run one atomic analytical question against the live context.

    Returns the raw answer text. The aggregator synthesizes across executors.
    """
    ctx_summary = json.dumps(_compact_context(context), default=str)[:2000]
    prompt = (
        "You are a focused trading analyst. Answer the single question below in "
        "3-6 concise sentences. Cite specific numbers from the context. Do not "
        "hedge — give your best directional read.\n\n"
        f"Question: {subtask}\n\nMarket context:\n{ctx_summary}"
    )
    res = await llm.complete(prompt, tier="fast", max_tokens=400, temperature=0.3)
    if res.error:
        return f"[fallback] no LLM available; {subtask} requires manual review."
    return res.text.strip()


def _compact_context(context: Dict) -> Dict:
    keep = (
        "ticker", "strike", "yes_ask", "yes_bid", "no_ask", "no_bid",
        "volume", "spot", "spot_1h_change", "spot_15m_change",
        "seconds_to_expiry", "market_implied_p", "realized_vol_15m",
        "distance_to_strike", "distance_to_strike_bps",
    )
    out: Dict = {}
    for k in keep:
        if k in context:
            out[k] = context[k]
    if "price_history" in context and isinstance(context["price_history"], list):
        out["price_history_tail"] = context["price_history"][-8:]
    return out
