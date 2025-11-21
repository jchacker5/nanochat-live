# Dockerfile for SRGI/NanoChat with EBM support
# Uses Python 3.10+ for THRML compatibility

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (needed for rustbpe)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy dependency files
COPY pyproject.toml uv.lock ./
COPY rustbpe/Cargo.toml rustbpe/Cargo.toml

# Install uv for fast package management (optional, can use pip)
RUN pip install --no-cache-dir uv

# Install Python dependencies
# Using uv for faster installs, but can fall back to pip
RUN uv pip install --system \
    torch>=2.8.0 \
    datasets>=4.0.0 \
    fastapi>=0.117.1 \
    files-to-prompt>=0.6 \
    psutil>=7.1.0 \
    regex>=2025.9.1 \
    setuptools>=80.9.0 \
    tiktoken>=0.11.0 \
    tokenizers>=0.22.0 \
    uvicorn>=0.36.0 \
    wandb>=0.21.3 \
    numpy \
    matplotlib \
    pytest>=8.0.0 \
    jax \
    jaxlib \
    equinox \
    scipy \
    || pip install --no-cache-dir \
    torch>=2.8.0 \
    datasets>=4.0.0 \
    fastapi>=0.117.1 \
    files-to-prompt>=0.6 \
    psutil>=7.1.0 \
    regex>=2025.9.1 \
    setuptools>=80.9.0 \
    tiktoken>=0.11.0 \
    tokenizers>=0.22.0 \
    uvicorn>=0.36.0 \
    wandb>=0.21.3 \
    numpy \
    matplotlib \
    pytest>=8.0.0 \
    jax \
    jaxlib \
    equinox \
    scipy

# Install THRML (optional, will work without it)
RUN pip install git+https://github.com/extropic-ai/thrml.git || echo "THRML installation failed, will use PyTorch fallback"

# Install optional dependencies
RUN pip install --no-cache-dir geoopt || echo "geoopt optional"

# Copy project files
COPY . .

# Build rustbpe extension
RUN cd rustbpe && maturin develop --release || echo "rustbpe build failed, may need manual build"

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["python", "-m", "scripts.chat_cli"]

