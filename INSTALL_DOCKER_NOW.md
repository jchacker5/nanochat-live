# Install Docker Desktop - Quick Steps

Docker Desktop installation requires your password. Here's what to do:

## Option 1: Complete Homebrew Installation (Recommended)

Run this command in your terminal (it will ask for your password):

```bash
brew install --cask docker
```

When prompted, enter your macOS password.

## Option 2: Manual Download & Install

1. **Download Docker Desktop**:
   ```bash
   open https://www.docker.com/products/docker-desktop/
   ```
   Or visit: https://www.docker.com/products/docker-desktop/

2. **Install**:
   - Download the `.dmg` file for Mac (Apple Silicon)
   - Open the downloaded file
   - Drag Docker to Applications
   - Launch Docker Desktop from Applications

3. **Complete Setup**:
   - Follow the setup wizard
   - Docker Desktop will start automatically

## After Installation

Once Docker Desktop is installed and running:

```bash
# Verify installation
docker --version
docker compose version

# Then build and start the container
cd /Users/jchacker5/Documents/nanochat-live
./docker-helper.sh build
./docker-helper.sh start
./docker-helper.sh shell
```

## Quick Test

After Docker Desktop is running, test it:

```bash
docker run hello-world
```

If this works, Docker is ready!

## Alternative: Use Setup Script (No Docker)

If you prefer not to install Docker right now, you can use the alternative setup:

```bash
./setup-without-docker.sh
```

This creates a Python virtual environment with all dependencies.

