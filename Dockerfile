# SIROM as a deployable service.
#
# Two stages so the wheels' build tooling never reaches the runtime image, and
# the runtime layer holds nothing but the interpreter, the installed packages
# and the source.
#
# Cold start is the number that matters when this runs scale-to-zero: importing
# ortools, scipy, scikit-learn and smt costs about a second on warm storage, so
# the container is ready in a few seconds and every solve after that is
# sub-second. `PYTHONDONTWRITEBYTECODE` is deliberately NOT set — precompiling
# to .pyc at build time is what keeps that first import near a second.

FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md ./
COPY sirom ./sirom
COPY demo ./demo

# The vrp extra carries httpx, which the routing surface imports. Without
# it that surface degrades to unavailable and /vrp/* returns 404.
RUN pip install --no-cache-dir '.[api,vrp]'

# Precompile everything now, so the first request does not pay for it.
RUN python -m compileall -q /opt/venv/lib/python3.11/site-packages || true


FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    # OpenBLAS and OMP spawn one thread per core by default and then fight each
    # other inside a 1-vCPU container. One thread each is measurably faster here.
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    # The routing surface reaches the solver over HTTP — it was built as a
    # separate service. In this combined image that is a loopback call to
    # ourselves; the default points at a docker-compose hostname that does not
    # exist here.
    SIROM_API_URL=http://127.0.0.1:8080

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY sirom ./sirom
COPY demo ./demo
COPY service.py ./service.py

# Run as a non-root user: this container accepts unauthenticated public input.
RUN useradd --create-home --uid 10001 sirom \
    && mkdir -p /home/sirom/.cache \
    && chown -R sirom:sirom /app /home/sirom
USER sirom

EXPOSE 8080

# Cloud Run injects PORT; the shell form expands it.
CMD exec uvicorn service:app --host 0.0.0.0 --port ${PORT} --workers 1 --log-level info
