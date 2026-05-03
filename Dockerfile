# ─────────────────────────────────────────────────────────────────────────────
# VehicleIQ — Dockerfile
# Python 3.11 slim | FastAPI + scikit-learn | No TensorFlow
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (minimal)
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY model/app.py              .
COPY model/model_artifacts/    model_artifacts/
COPY model/templates/          templates/
COPY model/static/             static/

EXPOSE 8000

# 2 workers is fine for t2/t3 micro; model loads fast with sklearn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--timeout-keep-alive", "30"]