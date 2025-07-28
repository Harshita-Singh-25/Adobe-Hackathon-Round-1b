FROM python:3.10-slim

WORKDIR /app

# Install system dependencies first
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Use pip's cache and faster PyPI mirror
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

COPY requirements.txt .

# Install Python dependencies with retries
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --retries 5 -r requirements.txt

COPY . .

CMD ["python", "solution.py"]