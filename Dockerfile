FROM python:3.11-slim

WORKDIR /app

COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

# App server and metadata — no sklearn/umap/numpy needed at runtime
COPY app.py ./
COPY datasets/meta.py datasets/meta.py
RUN touch datasets/__init__.py

# Static frontend
COPY static/ static/

# Mount point for the fly volume containing pre-computed embeddings
RUN mkdir -p data/embeddings

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
