# Multi-stage Dockerfile for AI Synopsis Flask
# Supports two build targets: web-server and refresh-cache

# Base stage with common dependencies
FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set common environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Web server stage
FROM base as web-server

# Install gunicorn for production web server
RUN pip install --no-cache-dir gunicorn

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api-status || exit 1

# Default command for web server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--threads", "4", "--timeout", "36000", "app:app"]

# Refresh cache stage
FROM base as refresh-cache

# Make the refresh script executable
RUN chmod +x /app/run_single_refresh.py

# Default command for single refresh
CMD ["python", "/app/run_single_refresh.py"]
