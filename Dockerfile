FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch first
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Pre-download Hugging Face model into the exact folder your code uses
RUN mkdir -p /models/cross-encoder/stsb-TinyBERT-L-4 && \
    python -c "from huggingface_hub import snapshot_download; \
               snapshot_download(repo_id='cross-encoder/stsb-TinyBERT-L-4', \
                                 local_dir='/models/cross-encoder/stsb-TinyBERT-L-4')"


# Copy your code
COPY . .

# Offline mode so no internet is needed at runtime
ENV TRANSFORMERS_OFFLINE=1

CMD ["python", "main.py"]
