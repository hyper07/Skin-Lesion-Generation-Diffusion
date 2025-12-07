FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 FASTAPI_APP=main.py

# Set the working directory
WORKDIR /app

# System dependencies required by some ML packages and general functionality
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       wget \
       git \
       ca-certificates \
       libsndfile1 \
       libasound2 \
       libglib2.0-0 \
       libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip/wheel first
RUN pip install --upgrade pip setuptools wheel

# Copy pyproject.toml first and install dependencies
COPY api/pyproject.toml /app/pyproject.toml

# Install the package and its dependencies
RUN pip install --no-cache-dir -e .

# Install additional requirements from requirements.txt
# COPY requirements.txt /app/requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy language model for word embeddings
RUN python -m spacy download en_core_web_lg

# Copy download scripts
COPY download_dataset.py /app/download_dataset.py
COPY download_models.py /app/download_models.py

# Copy the application source code
COPY . /app

EXPOSE 8888

# Default command: run uvicorn. Assumes the FastAPI app is available as `app` in `main.py`.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8888", "--reload"]