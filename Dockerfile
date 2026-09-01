FROM python:3.12-slim

WORKDIR /app

# System dependencies needed by Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

# Install CPU-only PyTorch first
RUN pip install --no-cache-dir torch==2.13.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app ./app
COPY static ./static

# Create local data directory
RUN mkdir -p data

# Expose FastAPI port
EXPOSE 8000

# Start the application
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
