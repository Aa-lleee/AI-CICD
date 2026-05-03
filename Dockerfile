FROM tensorflow/tensorflow:2.15.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libhdf5-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir gunicorn

# Copy model artifacts
COPY model/model_artifacts/     model_artifacts/

# Copy Flask app
COPY model/app.py               .
COPY model/templates/           templates/
COPY model/static/              static/

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120", "app:app"]