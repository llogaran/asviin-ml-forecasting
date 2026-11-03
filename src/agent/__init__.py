"""Natural-language assistant for forecasts and reorder recommendations (SPEC 4.8).

The public surface is intentionally tiny: :func:`answer` accepts a Spanish
question and an optional forecaster and returns the answer plus the raw data it
was built from. The CLI in :mod:`src.agent.cli` is a thin wrapper around it.
"""

from __future__ import annotations

from src.agent.agent import answer

__all__ = ["answer"]
