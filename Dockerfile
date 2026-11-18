FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/models

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
# Bake the ONNX models into the image so new pods start without downloading.
RUN python -m app.download_models

RUN useradd --create-home appuser && chown -R appuser /app /models
USER appuser

EXPOSE 8000
# One worker per container: scale with replicas (HPA), not with workers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
