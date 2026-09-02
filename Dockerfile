FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

COPY app ./app
COPY static ./static

RUN mkdir -p data

EXPOSE 8000

ENV OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
