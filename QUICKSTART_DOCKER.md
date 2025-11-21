# Quick Start: Docker Setup

## 🚀 Get Started in 3 Steps

### 1. Build the Docker Image

```bash
./docker-helper.sh build
```

This will:
- Download Python 3.11 base image
- Install all dependencies (PyTorch, JAX, THRML, etc.)
- Build rustbpe extension
- Set up the environment

**Time**: ~5-10 minutes (first time)

### 2. Start the Container

```bash
./docker-helper.sh start
```

### 3. Enter the Container

```bash
./docker-helper.sh shell
```

You're now inside a clean Python 3.11 environment with all dependencies!

## ✅ Verify Installation

Inside the container:

```bash
# Check Python version (should be 3.11)
python --version

# Test imports
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import jax; print('JAX:', jax.__version__)"
python -c "import thrml; print('THRML:', thrml.__version__)" || echo "THRML not installed (optional)"

# Test EBM module
python -c "from nanochat.ebm_hopfield import EBMHopfieldMemory; print('✓ EBM module works!')"

# Run tests
pytest tests/test_ebm_hopfield.py -v
```

## 🧹 Clean Up Local Dependencies (Optional)

If you want to remove packages from your system:

```bash
# Interactive cleanup script
./cleanup-local-deps.sh
```

**⚠️ Warning**: Only do this if you're sure you want to use Docker exclusively!

## 📝 Common Tasks

### Run Tests
```bash
./docker-helper.sh test
```

### Run EBM Tests
```bash
./docker-helper.sh ebm-test
```

### Install THRML (if needed)
```bash
./docker-helper.sh install-thrml
```

### View Logs
```bash
./docker-helper.sh logs
```

### Stop Container
```bash
./docker-helper.sh stop
```

## 🎯 Next Steps

1. **Start Training**:
   ```bash
   ./docker-helper.sh shell
   python -m scripts.base_train
   ```

2. **Generate Visualizations**:
   ```bash
   python scripts/visualize_srgi_phases.py
   ```

3. **Run EBM Experiments**:
   ```bash
   python -c "from nanochat.ebm_hopfield import EBMHopfieldMemory; from nanochat.ebm_trainer import PersistentEBMTrainer; print('Ready for EBM training!')"
   ```

## 💡 Tips

- **Code changes**: Edit files on your host machine, they're synced to the container
- **Data**: Put data in `./data/` directory (mounted read-only)
- **Checkpoints**: Saved to `./checkpoints/` (persists on host)
- **GPU**: Uncomment GPU section in `docker-compose.yml` if you have NVIDIA GPU

## 🆘 Troubleshooting

**Container won't start?**
```bash
docker-compose logs srgi
```

**Need to rebuild?**
```bash
./docker-helper.sh rebuild
```

**Permission issues?**
```bash
sudo chown -R $USER:$USER checkpoints outputs wandb
```

---

**You're all set!** Everything runs in Docker, keeping your system clean. 🎉

