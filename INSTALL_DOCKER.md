# Installing Docker

Docker is required to use the containerized setup. Here's how to install it:

## macOS Installation

### Option 1: Docker Desktop (Recommended)

1. **Download Docker Desktop**:
   - Visit: https://www.docker.com/products/docker-desktop/
   - Download for Mac (Apple Silicon or Intel)

2. **Install**:
   - Open the downloaded `.dmg` file
   - Drag Docker to Applications
   - Launch Docker Desktop
   - Complete the setup wizard

3. **Verify**:
   ```bash
   docker --version
   docker compose version
   ```

### Option 2: Homebrew

```bash
brew install --cask docker
```

Then launch Docker Desktop from Applications.

## After Installation

Once Docker is installed:

```bash
# Build the image
./docker-helper.sh build

# Start container
./docker-helper.sh start

# Enter container
./docker-helper.sh shell
```

## Alternative: Manual Setup (Without Docker)

If you prefer not to use Docker, you can set up a Python virtual environment:

```bash
# Create virtual environment with Python 3.10+
python3.10 -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Install THRML (optional, requires Python 3.10+)
pip install git+https://github.com/extropic-ai/thrml.git
```

## Troubleshooting

**Docker not starting?**
- Make sure Docker Desktop is running
- Check system requirements (macOS 10.15+)

**Permission denied?**
- Docker Desktop should handle permissions automatically
- May need to restart terminal after installation

**Still having issues?**
- Check Docker Desktop logs
- Ensure virtualization is enabled in system settings

