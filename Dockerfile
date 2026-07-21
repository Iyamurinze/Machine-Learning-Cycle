# syntax=docker/dockerfile:1
#
# API image for the malaria cell classification service.
#
# Multi-stage: dependencies are installed into a virtualenv in the builder,
# then only that venv is copied into the runtime image. This keeps pip, build
# caches and compiler toolchains out of the final layer.

FROM python:3.10-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copied alone so the dependency layer is cached and only reinstalls when
# requirements.txt itself changes, not on every source edit.
COPY requirements.txt .

# Streamlit, plotly and locust belong to the UI and load-test images, not the
# API. Stripping them here removes ~120 MB from the serving image.
RUN grep -viE "^(streamlit|plotly|locust|seaborn|matplotlib)" requirements.txt > api-requirements.txt \
    && pip install --no-cache-dir -r api-requirements.txt


FROM python:3.10-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=8000

# Keras writes to ~/.keras on import; give the non-root user a writable home.
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser api/ ./api/
COPY --chown=appuser:appuser models/ ./models/

# Upload staging and the data directory must be writable at runtime.
RUN mkdir -p data/uploads/Parasitized data/uploads/Uninfected data/train data/test \
    && chown -R appuser:appuser /app/data /app/models

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.getenv(\"PORT\",8000)}/health').read()" || exit 1

# A shell is needed so $PORT expands — Render and most PaaS platforms inject
# the port to bind rather than letting the image choose it. `exec` then
# replaces the shell with uvicorn so uvicorn becomes PID 1 and receives
# SIGTERM directly; without it the shell holds PID 1, swallows the signal, and
# every container stop waits out the full 10-second kill timeout.
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
