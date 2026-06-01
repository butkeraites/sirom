# SIROM HTTP API image.
#
# Multi-stage: most dependencies ship prebuilt manylinux wheels, but `smt` has
# no wheel for some platforms (e.g. linux/arm64) and compiles a C++ extension
# from source. The builder stage carries the toolchain and produces wheels for
# everything; the runtime stage installs those wheels into a clean slim image
# with no compiler, keeping it small and portable across architectures.

FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-api.txt ./
# Build/collect wheels for every dependency (compiles smt here).
RUN pip wheel --wheel-dir /wheels -r requirements-api.txt


FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies from the prebuilt wheels (no toolchain, no network).
COPY --from=builder /wheels /wheels
COPY requirements.txt requirements-api.txt ./
RUN pip install --no-index --find-links=/wheels -r requirements-api.txt \
    && rm -rf /wheels

# Install the package itself (deps already satisfied above).
COPY pyproject.toml README.md ./
COPY sirom ./sirom
RUN pip install --no-deps .

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

# Healthcheck via stdlib (slim has no curl).
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

# Single worker: the job store is in-process (the ProcessPoolExecutor still
# parallelizes solves). See sirom/api/jobs.py for the scale-out path.
CMD ["uvicorn", "sirom.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
