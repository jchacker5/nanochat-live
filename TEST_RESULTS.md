# NanoChat-Live Comprehensive Test Results

**Generated:** 2025-11-20 17:49:32

---

## nanochat.tokenizer

**Import Status:** ✅ Import successful

### Test Results

| Test | Status | Details |
|------|--------|---------|
| get_tokenizer | ✅ PASS | Success |
| encode/decode | ✅ PASS | Encoded and decoded: 'Hello, world!' == 'Hello, world!' |
| get_bos_token_id | ✅ PASS | BOS token ID: 256 |


## nanochat.gpt

**Import Status:** ✅ Import successful

### Test Results

| Test | Status | Details |
|------|--------|---------|
| GPTConfig creation | ✅ PASS | Config created successfully |
| GPT model creation | ✅ PASS | Model created successfully |


## nanochat.engine

**Import Status:** ✅ Import successful

### Test Results

| Test | Status | Details |
|------|--------|---------|
| KVCache creation | ✅ PASS | KVCache created successfully |
| KVCache resize | ❌ FAIL | Resized from 128 to 128 |


## nanochat.ssm

**Import Status:** ✅ Import successful

### Test Results

| Test | Status | Details |
|------|--------|---------|
| StableResonantSSM creation | ✅ PASS | SSM created successfully |
| SSM forward pass | ✅ PASS | Output shape: torch.Size([2, 10, 64]) |
| ResonantBlock forward pass | ✅ PASS | Block output shape: torch.Size([2, 10, 64]) |


## nanochat.checkpoint_manager

**Import Status:** ✅ Import successful

### Test Results

| Test | Status | Details |
|------|--------|---------|
| get_base_dir | ✅ PASS | Base dir: /Users/jchacker5/.cache/nanochat |


## nanochat.common

**Import Status:** ✅ Import successful

### Test Results

| Test | Status | Details |
|------|--------|---------|
| get_base_dir | ✅ PASS | Base dir: /Users/jchacker5/.cache/nanochat |
| autodetect_device_type | ✅ PASS | Device type: mps |


## nanochat.dataset

**Import Status:** ✅ Import successful

### Test Results

| Test | Status | Details |
|------|--------|---------|
| list_parquet_files | ✅ PASS | Found 1 parquet files |


## tasks

**Import Status:** ✅ Tasks module

### Test Results

| Test | Status | Details |
|------|--------|---------|
| tasks.arc | ✅ PASS | ARC imported successfully |
| tasks.gsm8k | ✅ PASS | GSM8K imported successfully |
| tasks.mmlu | ✅ PASS | MMLU imported successfully |
| tasks.humaneval | ✅ PASS | HumanEval imported successfully |
| tasks.smoltalk | ✅ PASS | SmolTalk imported successfully |


---

## Summary

- **Total Tests:** 19
- **Passed:** 18 ✅
- **Failed:** 1 ❌
- **Success Rate:** 94.7%