# 🚀 Quick Start: Google Colab (2 Minutes)

## Step 1: Sign In to Google Colab

1. Go to: **https://colab.research.google.com/**
2. Sign in with your Google account
3. That's it! You're ready.

## Step 2: Open the Notebook

**Option A: Upload from GitHub**
1. In Colab, click **File** → **Upload notebook**
2. Go to: **https://github.com/jchacker5/nanochat-live/blob/master/colab_setup.ipynb**
3. Click **Raw** button (top right)
4. Copy the URL
5. In Colab: **File** → **Open notebook** → Paste URL

**Option B: Direct Link**
1. Click this link: [Open in Colab](https://colab.research.google.com/github/jchacker5/nanochat-live/blob/master/colab_setup.ipynb)
2. It will open directly in Colab!

## Step 3: Enable GPU

**IMPORTANT - Do this first!**

1. Click **Runtime** → **Change runtime type**
2. Set **Hardware accelerator** to **GPU (T4)**
3. Click **Save**

## Step 4: Run All Cells

1. Click **Runtime** → **Run all**
2. Or press `Ctrl+F9` (Windows) / `Cmd+F9` (Mac)
3. Wait ~30-60 minutes
4. Done! ✅

---

## What Happens Automatically

The notebook will:
- ✅ Check GPU availability
- ✅ Clone your repository
- ✅ Install all dependencies
- ✅ Run theory validation tests
- ✅ Run EBM experiments
- ✅ (Optional) Train a small model

**You don't need to do anything else!**

---

## Troubleshooting

**"No GPU available"?**
- Make sure you enabled GPU in Step 3
- Free tier has usage limits - wait or upgrade

**"Out of memory"?**
- Reduce batch size in training cell
- Or skip training, just run theory tests

**"Connection lost"?**
- Colab free tier disconnects after ~90 min
- Re-run cells (they're idempotent)
- Or upgrade to Colab Pro

---

## Results

After completion, you'll have:
- ✅ Theory validation results
- ✅ EBM experiment results  
- ✅ (Optional) Trained model checkpoints

All results are saved in the notebook and can be downloaded!

---

**That's it! Just sign in, enable GPU, and run all cells.** 🎉

