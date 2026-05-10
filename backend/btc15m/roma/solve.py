"""Recursive ROMA solve loop.

Mirrors the TS pseudocode:

    solve(goal, context):
      if atomizer.isAtomic(goal):
          return executor.run(goal, context)
      else:
          subtasks = planner.decompose(goal, context)
          results  = await Promise.all(subtasks.map(t => solve(t, context)))
          return aggregator.synthesize(results, context)
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from ..llm import Llm
from .atomizer import is_atomic
from .planner import generate_subtasks
from .executor import execute_subtask
from .aggregator import synthesize

logger = logging.getLogger("trading_bot")


@dataclass
class RomaResult:
    thesis: str
    p_yes: float
    recommendation: str  # BUY_YES | BUY_NO | PASS
    confidence: float
    key_drivers: List[str] = field(default_factory=list)
    subtasks: List[Tuple[str, str]] = field(default_factory=list)
    elapsed_ms: float = 0.0


async def solve(goal: str, context: Dict, llm: Llm | None = None) -> RomaResult:
    """Top-level entry point. Always decomposes for production trading goals."""
    llm = llm or Llm()
    start = time.monotonic()

    # Stage 1 — atomizer (kept for fidelity; production goals always decompose).
    atomic = await is_atomic(llm, goal, context)
    if atomic:
        answer = await execute_subtask(llm, goal, context)
        agg = await synthesize(llm, goal, [(goal, answer)], context)
        return _build_result(agg, [(goal, answer)], start)

    # Stage 2 — planner.
    subtasks = await generate_subtasks(llm, goal, context)
    logger.info(f"ROMA planned {len(subtasks)} subtasks")

    # Stage 3 — executors in parallel.
    answers = await asyncio.gather(*[execute_subtask(llm, t, context) for t in subtasks])
    pairs = list(zip(subtasks, answers))

    # Stage 4 — aggregator (with structured extraction).
    agg = await synthesize(llm, goal, pairs, context)
    return _build_result(agg, pairs, start)


def _build_result(agg: Dict, pairs: List[Tuple[str, str]], start: float) -> RomaResult:
    return RomaResult(
        thesis=agg["thesis"],
        p_yes=agg["p_yes"],
        recommendation=agg["recommendation"],
        confidence=agg["confidence"],
        key_drivers=agg.get("key_drivers", []),
        subtasks=pairs,
        elapsed_ms=(time.monotonic() - start) * 1000,
    )
