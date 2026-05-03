FROM python:3.10-slim

WORKDIR /app

# Pin TensorFlow to a stable version that matches your training environment
RUN pip install tensorflow==2.15.0 keras==2.15.0

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY model/ ./model/
COPY app.py .

EXPOSE 8000

CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:8000", "--timeout", "120", "app:app"]