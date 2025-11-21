# Docker Setup for SRGI/NanoChat

This guide explains how to use Docker to containerize the SRGI project, keeping your system clean while having all dependencies isolated.

## Quick Start

```bash
# Build the image
./docker-helper.sh build

# Start the container
./docker-helper.sh start

# Enter the container shell
./docker-helper.sh shell

# Run tests
./docker-helper.sh test
```

## Prerequisites

- Docker installed and running
- Docker Compose installed

## Features

- **Python 3.11** (supports THRML)
- **All dependencies** pre-installed
- **GPU support** (optional, uncomment in docker-compose.yml)
- **Volume mounts** for development
- **Jupyter notebook** service (optional)

## Usage

### Using the Helper Script

```bash
# Build image
./docker-helper.sh build

# Start container
./docker-helper.sh start

# Enter container
./docker-helper.sh shell

# Run tests
./docker-helper.sh test

# Run EBM tests
./docker-helper.sh ebm-test

# Install THRML (if needed)
./docker-helper.sh install-thrml

# View logs
./docker-helper.sh logs

# Stop container
./docker-helper.sh stop

# Clean up everything
./docker-helper.sh clean
```

### Using Docker Compose Directly

```bash
# Build and start
docker-compose up -d

# Enter container
docker-compose exec srgi /bin/bash

# Run a command
docker-compose exec srgi python -m pytest tests/

# Stop
docker-compose down
```

## Inside the Container

Once inside the container shell:

```bash
# Verify Python version (should be 3.11)
python --version

# Check THRML installation
python -c "import thrml; print('THRML:', thrml.__version__)"

# Run EBM tests
pytest tests/test_ebm_hopfield.py -v

# Start training
python -m scripts.base_train

# Generate visualizations
python scripts/visualize_srgi_phases.py
```

## Volume Mounts

The following directories are mounted:

- **`/app`**: Project root (your code)
- **`/app/data`**: Data directory (read-only)
- **`/app/checkpoints`**: Model checkpoints
- **`/app/outputs`**: Training outputs
- **`/app/wandb`**: Weights & Biases logs

## GPU Support

To enable GPU support, uncomment the GPU section in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

Then rebuild:

```bash
docker-compose build --no-cache
```

## Jupyter Notebook

Start the Jupyter service:

```bash
docker-compose up -d jupyter
```

Access at: http://localhost:8888

## Troubleshooting

### Container won't start
```bash
# Check logs
docker-compose logs srgi

# Rebuild from scratch
./docker-helper.sh rebuild
```

### THRML not working
```bash
# Install manually
docker-compose exec srgi pip install git+https://github.com/extropic-ai/thrml.git
```

### Rust build fails
```bash
# Build rustbpe manually inside container
docker-compose exec srgi bash
cd rustbpe
maturin develop --release
```

### Permission issues
```bash
# Fix ownership
sudo chown -R $USER:$USER checkpoints outputs wandb
```

## Cleaning Up

Remove all containers and images:

```bash
./docker-helper.sh clean
```

Or manually:

```bash
docker-compose down -v
docker rmi nanochat-srgi
```

## Benefits

✅ **Clean system**: No Python packages installed on your machine  
✅ **Reproducible**: Same environment for everyone  
✅ **Isolated**: Dependencies don't conflict  
✅ **Portable**: Works on any machine with Docker  
✅ **GPU ready**: Easy GPU support when needed  

## Next Steps

1. Build the image: `./docker-helper.sh build`
2. Start container: `./docker-helper.sh start`
3. Enter shell: `./docker-helper.sh shell`
4. Start training!

