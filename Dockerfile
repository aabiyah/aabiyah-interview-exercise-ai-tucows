FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Build FAISS index on container build
RUN python scripts/build_index.py

# Expose port
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["sh", "-c", "PYTHONPATH=src python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000"]
