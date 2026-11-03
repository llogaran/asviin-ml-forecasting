"""Monitoring package: distribution-drift checks for the demand dataset.

``detect_drift`` is re-exported lazily so ``python -m src.monitoring.drift`` does
not import the CLI module before ``runpy`` executes it.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.monitoring.drift import detect_drift

__all__ = ["detect_drift"]


def __getattr__(name: str) -> Any:
    if name == "detect_drift":
        from src.monitoring.drift import detect_drift

        return detect_drift
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
