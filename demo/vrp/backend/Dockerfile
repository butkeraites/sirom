# VRP demo backend: parses VRP-REP datasets and talks to the SIROM HTTP API.
# Light image — no ortools/smt/sirom; just FastAPI + an HTTP client.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY app.py vrprep.py cvrp.py ./

RUN useradd --create-home --uid 10002 appuser
USER appuser

EXPOSE 8801

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8801/health').status==200 else 1)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8801"]
