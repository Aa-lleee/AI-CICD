# ── Stage: Runtime ────────────────────────────────────────────────────────────
# Use slim Python 3.10 — TensorFlow 2.x requires 3.8–3.11
FROM python:3.10-slim

# Keeps Python from writing .pyc files and buffers stdout/stderr immediately
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps needed by TensorFlow / h5py
RUN apt-get update && apt-get install -y --no-install-recommends \
        libhdf5-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Copy application source ───────────────────────────────────────────────────
# Model artifacts (pre-trained — no GPU needed at runtime)
COPY model_artifacts/           model_artifacts/

# Flask app + HTML template
COPY app.py                     .
COPY templates/index.html       templates/index.html

# Vehicle images served as static files
COPY bike.png hatchback.png sedan.png suv.png truck.png \
     static/images/

# ── Expose & run ──────────────────────────────────────────────────────────────
EXPOSE 8000

# Use gunicorn for production instead of Flask dev server
RUN pip install --no-cache-dir gunicorn

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", \
     "--timeout", "120", "app:app"]