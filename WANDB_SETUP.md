# 🔑 Wandb Setup for Colab Training

## Secure Setup (Recommended)

**Don't share your API key with me!** Use Colab's built-in secrets instead.

### Step 1: Get Your Wandb API Key

1. Go to https://wandb.ai/settings
2. Copy your API key (starts with something like `abc123...`)

### Step 2: Add to Colab Secrets

1. In Colab, click the **🔑 (key icon)** in the left sidebar
2. Click **"Secrets"** tab
3. Click **"+ Add secret"**
4. Name: `WANDB_API_KEY`
5. Value: Paste your API key
6. Click **"Add secret"**

### Step 3: Run the Notebook

The notebook will automatically:
- ✅ Detect the secret
- ✅ Log in to wandb
- ✅ Track your training run

## What Happens

- **With API key**: Training logs to wandb (project: `nanochat`, run: `srgi-production-a100`)
- **Without API key**: Training uses `--run=dummy` (no logging, still works!)

## Viewing Results

Once training starts:
1. Go to https://wandb.ai
2. Open project: `nanochat`
3. Find run: `srgi-production-a100`
4. See real-time training metrics!

## Security Notes

- ✅ **DO**: Use Colab secrets (secure, hidden)
- ❌ **DON'T**: Paste API key directly in notebook cells
- ❌ **DON'T**: Share your API key with anyone

---

**The notebook is already configured! Just add your API key to Colab secrets and run.**

