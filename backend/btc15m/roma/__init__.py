"""ROMA (Recursive Open Meta-Agent) engine — Python port.

Architecture (ported from sentient-market-reader/lib/roma):
  Atomizer  [fast]  — atomic or decompose?
  Planner   [smart] — generate 3-5 subtasks
  Executors [fast]  — Promise.all(subtasks)
  Aggregator[smart] — synthesize into a unified thesis
"""
from .solve import solve, RomaResult  # noqa: F401
