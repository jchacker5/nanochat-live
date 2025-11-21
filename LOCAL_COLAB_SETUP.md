# Local Jupyter + Colab Setup

This setup allows you to use Colab's interface while running code on your local Mac.

## ⚠️ Important Notes

**GPU Limitation**: Your Mac doesn't have CUDA GPU, so:
- ✅ Great for: Development, testing, theory validation (CPU)
- ❌ Not for: GPU-accelerated training (use cloud GPUs instead)
- 💡 Best use: Develop/test locally, train on cloud

## Setup Steps

### 1. Install Dependencies

```bash
# Use python3 -m pip if pip not in PATH
python3 -m pip install jupyter jupyter_http_over_ws

# OR if you have pip directly
pip install jupyter jupyter_http_over_ws
```

### 2. Enable Extension

```bash
# For Jupyter Notebook (older)
jupyter serverextension enable --py jupyter_http_over_ws

# OR for Jupyter Server (newer)
jupyter server extension enable --py jupyter_http_over_ws
```

### 3. Start Local Jupyter Server

```bash
# Use the helper script
./start_local_jupyter.sh

# OR manually:
jupyter notebook \
    --NotebookApp.allow_origin='https://colab.research.google.com' \
    --port=8888 \
    --NotebookApp.port_retries=0 \
    --no-browser \
    --notebook-dir="$(pwd)"
```

### 4. Connect from Colab

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `colab_setup.ipynb` or create new notebook
3. Click **"Connect"** button (top right)
4. Select **"Connect to local runtime..."**
5. Paste the URL from step 3 (includes token)
6. Click **"Connect"**

## Usage

### For Development/Testing (Local)
- ✅ Run theory validation tests
- ✅ Test EBM experiments  
- ✅ Develop new features
- ✅ Debug code
- ⚠️ Slow for training (CPU only)

### For Training (Cloud GPU)
- Use Colab with GPU runtime (not local)
- Or use Google Cloud Notebooks
- See `QUICK_CLOUD_START.md` for details

## Troubleshooting

**Extension not found?**
```bash
python3 -m pip install --upgrade jupyter jupyter_http_over_ws
python3 -m jupyter server extension list  # Check if enabled
```

**Port 8888 already in use?**
```bash
# Use different port
jupyter notebook --port=8889 --NotebookApp.allow_origin='https://colab.research.google.com'
```

**Connection refused?**
- Make sure firewall allows port 8888
- Check that Jupyter is running
- Verify the token in the URL

## Recommended Workflow

1. **Develop locally** (this setup)
   - Write/test code
   - Run theory validation
   - Debug issues

2. **Train on cloud** (Colab GPU or Cloud Notebooks)
   - Upload to Colab with GPU
   - Or use `cloud_train.sh` on cloud instance
   - Get GPU acceleration

3. **Analyze results locally**
   - Download checkpoints
   - Run evaluation
   - Generate visualizations

---

**Best of both worlds**: Colab UI + Local development + Cloud training! 🚀

