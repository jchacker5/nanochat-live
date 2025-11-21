# NanoChat-Live Comprehensive Test Results

**Generated:** 2025-11-20 21:26:20

---

## nanochat.tokenizer

**Import Status:** ❌ No module named 'tokenizers'



## nanochat.gpt

**Import Status:** ❌ No module named 'torch'



## nanochat.engine

**Import Status:** ❌ No module named 'torch'



## nanochat.ssm

**Import Status:** ❌ No module named 'torch'



## nanochat.checkpoint_manager

**Import Status:** ❌ No module named 'torch'



## nanochat.common

**Import Status:** ❌ No module named 'torch'



## nanochat.dataset

**Import Status:** ❌ No module named 'requests'



## tasks

**Import Status:** ✅ Tasks module

### Test Results

| Test | Status | Details |
|------|--------|---------|
| tasks.arc | ❌ FAIL | No module named 'datasets' |
| tasks.gsm8k | ❌ FAIL | No module named 'datasets' |
| tasks.mmlu | ❌ FAIL | No module named 'datasets' |
| tasks.humaneval | ❌ FAIL | No module named 'datasets' |
| tasks.smoltalk | ❌ FAIL | No module named 'datasets' |


## scripts

**Import Status:** ✅ Scripts module

### Test Results

| Test | Status | Details |
|------|--------|---------|
| scripts.base_train | ❌ FAIL | Import error: No module named 'torch' |
| scripts.chat_cli | ❌ FAIL | Import error: No module named 'torch' |
| scripts.chat_web | ❌ FAIL | Import error: No module named 'torch' |
| scripts.chat_sft | ❌ FAIL | Import error: No module named 'torch' |
| scripts.ssm_demo | ❌ FAIL | Import error: No module named 'torch' |
| scripts.tok_train | ❌ FAIL | Import error: No module named 'torch' |
| scripts.tok_eval | ❌ FAIL | Import error: No module named 'tokenizers' |


---

## Summary

- **Total Tests:** 12
- **Passed:** 0 ✅
- **Failed:** 12 ❌
- **Success Rate:** 0.0%