"""CLI for the natural-language assistant (SPEC 4.8).

Usage::

    .venv/bin/python -m src.agent.cli "¿Cuánto debo pedir de la hojuela de PET?"

Only the Spanish answer is printed on success; errors go to stderr and the
process exits non-zero so Makefiles and CI can detect failures.
"""

from __future__ import annotations

import argparse
import sys

from src.agent.agent import answer


def main(argv: list[str] | None = None) -> int:
    """Parse a question, print the answer and return the process exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m src.agent.cli",
        description="Ask the Asviin forecasting assistant a question in Spanish.",
    )
    parser.add_argument(
        "question",
        nargs="+",
        help="Question in Spanish, e.g. '¿Cuánto debo pedir de la hojuela de PET?'",
    )
    args = parser.parse_args(argv)
    question = " ".join(args.question).strip()
    try:
        result = answer(question)
    except Exception as exc:  # noqa: BLE001 - CLI must report every failure to the user
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(result["answer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
