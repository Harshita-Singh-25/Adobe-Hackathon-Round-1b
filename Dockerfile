# Use a smaller base image with Python 3.9
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install system dependencies needed for PyMuPDF
RUN apt-get update && \
    apt-get install -y --no-install-recommends libfreetype6 && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies with no cache
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Run the application
CMD ["python", "adobe_hackathon_solution.py"]