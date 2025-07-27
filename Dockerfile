# Dockerfile
FROM --platform=linux/amd64 python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main application
COPY adobe_hackathon_solution.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Default command
CMD ["python", "adobe_hackathon_solution.py"]

# requirements.txt content:
# PyMuPDF==1.23.26
# scikit-learn==1.3.2
# numpy==1.24.4
# pandas==2.0.3