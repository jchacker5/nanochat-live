# Quick Cloud Training Guide

## 🚀 Fastest Way: Google Colab (5 minutes)

### Step 1: Open Colab Notebook
1. Go to [Google Colab](https://colab.research.google.com/)
2. File > Upload notebook
3. Upload `colab_setup.ipynb` from this repo
4. **Enable GPU**: Runtime > Change runtime type > GPU (T4)

### Step 2: Run All Cells
- Click "Runtime > Run all"
- Wait ~10-15 minutes for setup
- Training will start automatically

### Step 3: Monitor Progress
- Check cell outputs for training logs
- Checkpoints saved in `checkpoints/` folder
- Download when done

**Cost**: Free (with usage limits) or ~$10/month for Colab Pro

---

## ☁️ Better Performance: Google Cloud Notebooks

### Step 1: Create Instance
```bash
# In Google Cloud Console:
# 1. Go to Vertex AI > Workbench
# 2. Create new notebook
# 3. Choose GPU: T4 or A100
# 4. Python 3 environment
```

### Step 2: Clone & Run
```bash
git clone https://github.com/jchacker5/nanochat-live.git
cd nanochat-live
chmod +x cloud_train.sh
./cloud_train.sh
```

**Cost**: ~$0.50-1.00/hour for T4, ~$3-4/hour for A100

---

## 🤗 Alternative: Hugging Face Spaces

### For Inference/Demos Only
1. Create new Space on Hugging Face
2. Use GPU instance
3. Upload your code
4. Deploy

**Note**: Not ideal for training, better for inference

---

## 📊 What You'll Validate

After training completes, you'll have:

1. ✅ **Theory Validation Results**
   - Resonance stability confirmed
   - Phase synchronization working
   - Geometric structure validated

2. ✅ **Trained Model**
   - Checkpoints in `checkpoints/`
   - Ready for evaluation

3. ✅ **EBM Experiments**
   - Energy-based memory working
   - Attractor basins confirmed

---

## 🎯 Recommended Training Configs

### Quick Validation (30-60 min on T4)
```bash
python -m scripts.base_train \
    --depth=8 \
    --max_seq_len=2048 \
    --device_batch_size=8 \
    --total_batch_size=16384 \
    --num_iterations=500 \
    --run=srgi-quick-test
```

### Full Validation (2-4 hours on T4)
```bash
python -m scripts.base_train \
    --depth=20 \
    --max_seq_len=2048 \
    --device_batch_size=16 \
    --total_batch_size=65536 \
    --num_iterations=2000 \
    --run=srgi-full-validation
```

### Production Training (8+ hours on A100)
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train \
    --depth=20 \
    --max_seq_len=2048 \
    --device_batch_size=32 \
    --total_batch_size=524288 \
    --num_iterations=10000 \
    --run=srgi-production
```

---

## 💡 Tips

1. **Start Small**: Use depth=8 for initial validation
2. **Monitor GPU**: Watch VRAM usage, adjust batch size
3. **Save Checkpoints**: They're automatically saved
4. **Use WandB**: Set `--run=your-experiment-name` to track

---

## 🆘 Troubleshooting

**Out of Memory?**
- Reduce `device_batch_size`
- Reduce `max_seq_len`
- Use smaller model (`--depth=4`)

**Too Slow?**
- Use A100 instead of T4
- Reduce `num_iterations` for quick test
- Use mixed precision (already enabled)

**Tokenizer Issues?**
- Make sure Rust is installed
- Rebuild: `cd rustbpe && maturin build --release`

---

**Ready to validate your theories! 🚀**

