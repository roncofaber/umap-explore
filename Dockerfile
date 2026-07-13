FROM python:3.11-slim

WORKDIR /app

COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

# App server and metadata — no sklearn/umap/numpy needed at runtime
COPY app.py utils.py ./
COPY datasets/meta.py datasets/meta.py
RUN touch datasets/__init__.py

# Static frontend and pre-computed embeddings
COPY static/ static/
COPY data/embeddings/ data/embeddings/

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
