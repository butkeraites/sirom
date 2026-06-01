# SIROM HTTP API image.
# Single stage: every dependency ships a prebuilt manylinux wheel, so there is
# no build toolchain to discard and multi-stage would save little.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first so this layer caches across code changes.
COPY requirements.txt requirements-api.txt ./
RUN pip install -r requirements-api.txt

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
