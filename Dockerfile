# Dockerfile for Demand Forecasting and Price Optimization Service

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all main scripts
COPY pipeline.py .
COPY train_with_pipeline.py .
COPY predict_with_pipeline.py .
COPY evaluate_model.py .

# Copy source code
COPY src/ ./src/

# Copy configuration files
COPY constraints*.yaml ./
COPY Makefile .

# Copy data files
COPY retail_pricing_demand_2024.csv ./
COPY simcel-6pk70-1jk5iqdp-train_v9rqX0R.csv ./
COPY retail_pricing_demand_2024_sample.csv ./

# Create necessary directories with proper permissions
RUN mkdir -p models reports notebooks tests data config \
    && chmod -R 755 models reports notebooks tests data config

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port for potential API services
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.append('src'); from data_loader import DataLoader; print('OK')" || exit 1

# Default command - show pipeline help
CMD ["python", "pipeline.py", "--help"]