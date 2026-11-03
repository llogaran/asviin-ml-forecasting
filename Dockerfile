# syntax=docker/dockerfile:1

# Serving image for the Asviin ML forecasting API.
#
# The model is trained offline (`make train`) and its artifacts are committed to
# the repository; the image only installs the package and serves the committed
# model. No training happens at build time.
#
#   docker build -t asviin-ml-forecasting:local .
#   docker run --rm -p 8000:8000 asviin-ml-forecasting:local

# ---------------------------------------------------------------------------
# Builder: resolve and build every wheel once, with runtime deps only.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Only the build inputs are copied so the wheel layer is invalidated by
# packaging/source changes, never by data, artifacts or docs.
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip wheel --wheel-dir /wheels .

# ---------------------------------------------------------------------------
# Runtime: slim, non-root image with the committed dataset and model artifacts.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MLFC_ENV=production \
    PYTHONPATH=/app

WORKDIR /app

# Dependencies plus the project wheel, installed fully offline from the builder.
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-index --find-links=/wheels asviin-ml-forecasting \
    && rm -rf /wheels

# Dedicated unprivileged account; nothing in /app is written at runtime.
RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid appuser --create-home --shell /usr/sbin/nologin appuser

# The dataset and artifacts are committed deliverables (synthetic data only).
COPY data ./data
COPY artifacts ./artifacts
# Kept next to the installed package so src.config.PROJECT_ROOT resolves to /app.
COPY src ./src
COPY README.md ./

USER appuser

EXPOSE 8000

# Liveness only: /health never needs the model or an API key.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
