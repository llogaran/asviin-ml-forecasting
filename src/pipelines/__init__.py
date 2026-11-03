"""Data, modeling and serving pipelines for Asviin ML Forecasting.

Each module exposes a small public API plus a ``python -m`` CLI so the Makefile
can run the whole chain without importing private helpers.
"""

import warnings

# Some sandboxed/dev environments cannot access /dev/shm; joblib then prints a
# warning that embeds an absolute path. It is environmental noise, not a model
# issue, and it would leak local paths into CLI/API output.
warnings.filterwarnings("ignore", message=".*joblib will operate in serial mode.*")
