# Dockerfile for Face Recognition App on NVIDIA Jetson Nano
# Base image with CUDA, TensorFlow, and OpenCV optimized for Jetson
FROM nvcr.io/nvidia/l4t-ml:r32.7.1-py3

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-yaml \
    && rm -rf /var/lib/apt/lists/*

# Fix for distutils in older L4T environments
RUN pip3 install --upgrade pip setuptools wheel


# Copy requirements and install remaining Python dependencies
# We skip tensorflow and keras in requirements.txt since they are pre-installed
# or handled via TFLite fallback.
COPY req_jetson.txt .
RUN pip3 install --no-cache-dir -r req_jetson.txt

# Copy application files
COPY config/ ./config/
COPY src/ ./src/
COPY examples/ ./examples/

# Create directories for models and data
RUN mkdir -p models data

# Set environment variables
ENV PYTHONPATH=/app/src:$PYTHONPATH
ENV TF_CPP_MIN_LOG_LEVEL=2

# Default command
CMD ["python3", "src/face_recognition_app.py"]
