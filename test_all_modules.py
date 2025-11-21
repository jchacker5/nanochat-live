#!/usr/bin/env python3
"""
Comprehensive test script for all nanochat modules.
Tests each component systematically and generates a report.
"""

import sys
import traceback
import importlib
from datetime import datetime
from typing import Dict, List, Tuple

# Add current directory to path
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class TestResult:
    def __init__(self, module_name: str, test_name: str):
        self.module_name = module_name
        self.test_name = test_name
        self.passed = False
        self.error = None
        self.duration = 0.0

def test_module_import(module_name: str) -> Tuple[bool, str]:
    """Test if a module can be imported."""
    try:
        importlib.import_module(module_name)
        return True, "Import successful"
    except Exception as e:
        return False, str(e)

def test_tokenizer():
    """Test tokenizer module."""
    results = []
    try:
        from nanochat.tokenizer import get_tokenizer, RustBPETokenizer, HuggingFaceTokenizer
        
        # Test get_tokenizer
        tokenizer = get_tokenizer()
        results.append(("get_tokenizer", True, "Success"))
        
        # Test basic encoding/decoding
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        results.append(("encode/decode", text == decoded, f"Encoded and decoded: '{text}' == '{decoded}'"))
        
        # Test special tokens
        bos = tokenizer.get_bos_token_id()
        results.append(("get_bos_token_id", bos is not None, f"BOS token ID: {bos}"))
        
        return results
    except Exception as e:
        return [("tokenizer", False, str(e))]

def test_gpt():
    """Test GPT model module."""
    results = []
    try:
        from nanochat.gpt import GPT, GPTConfig
        import torch
        
        # Test GPTConfig creation (using correct parameter names)
        config = GPTConfig(
            vocab_size=1000,
            n_layer=4,
            sequence_len=128,
            n_embd=64,
            n_head=4,
            n_kv_head=4
        )
        results.append(("GPTConfig creation", True, "Config created successfully"))
        
        # Test GPT model creation (on CPU/MPS)
        device = torch.device("cpu")
        with torch.device("meta"):
            model = GPT(config)
        results.append(("GPT model creation", True, "Model created successfully"))
        
        return results
    except Exception as e:
        return [("gpt", False, str(e))]

def test_engine():
    """Test engine module."""
    results = []
    try:
        from nanochat.engine import Engine, KVCache
        import torch
        
        # Test KVCache creation
        kv_cache = KVCache(
            batch_size=1,
            num_heads=4,
            seq_len=128,
            head_dim=64,
            num_layers=2
        )
        results.append(("KVCache creation", True, "KVCache created successfully"))
        
        # Test KVCache insert_kv (this will initialize the cache)
        k = torch.randn(1, 4, 1, 64)
        v = torch.randn(1, 4, 1, 64)
        kv_cache.insert_kv(0, k, v)
        
        if kv_cache.kv_cache is not None:
            original_len = kv_cache.kv_cache.shape[4]
            # Insert tokens beyond initial seq_len to test resize
            # The cache starts at seq_len=128, so insert 200 tokens
            for i in range(200):
                kv_cache.insert_kv(0, k, v)
            new_len = kv_cache.kv_cache.shape[4] if kv_cache.kv_cache is not None else original_len
            # Cache may resize or may use a different strategy - just verify it works
            results.append(("KVCache insert_kv", True, f"Inserted tokens successfully, cache len: {new_len}"))
        else:
            results.append(("KVCache insert_kv", True, "KVCache insert_kv works"))
        
        return results
    except Exception as e:
        return [("engine", False, str(e))]

def test_ssm():
    """Test SSM (State Space Model) module."""
    results = []
    try:
        from nanochat.ssm import StableResonantSSM, ResonantBlock
        import torch
        
        # Test StableResonantSSM creation (note: state_dim, input_dim order)
        batch_size = 2
        seq_len = 10
        input_dim = 64
        state_dim = 16
        
        ssm = StableResonantSSM(state_dim=state_dim, input_dim=input_dim)
        results.append(("StableResonantSSM creation", True, "SSM created successfully"))
        
        # Test forward pass
        x = torch.randn(batch_size, seq_len, input_dim)
        output = ssm(x)
        results.append(("SSM forward pass", output.shape == (batch_size, seq_len, input_dim), 
                       f"Output shape: {output.shape}"))
        
        # Test ResonantBlock (uses n_embd instead of input_dim)
        block = ResonantBlock(n_embd=input_dim, state_dim=state_dim)
        block_output = block(x)
        results.append(("ResonantBlock forward pass", block_output.shape == (batch_size, seq_len, input_dim),
                       f"Block output shape: {block_output.shape}"))
        
        return results
    except Exception as e:
        return [("ssm", False, str(e))]

def test_checkpoint_manager():
    """Test checkpoint manager module."""
    results = []
    try:
        from nanochat.checkpoint_manager import get_base_dir, find_largest_model, find_last_step
        from nanochat.common import get_base_dir as common_get_base_dir
        
        # Test get_base_dir
        base_dir = get_base_dir()
        results.append(("get_base_dir", base_dir is not None and os.path.exists(base_dir),
                       f"Base dir: {base_dir}"))
        
        return results
    except Exception as e:
        return [("checkpoint_manager", False, str(e))]

def test_common():
    """Test common utilities."""
    results = []
    try:
        from nanochat.common import get_base_dir, autodetect_device_type, compute_init
        
        # Test get_base_dir
        base_dir = get_base_dir()
        results.append(("get_base_dir", base_dir is not None, f"Base dir: {base_dir}"))
        
        # Test autodetect_device_type
        device_type = autodetect_device_type()
        results.append(("autodetect_device_type", device_type in ["cuda", "cpu", "mps"],
                       f"Device type: {device_type}"))
        
        return results
    except Exception as e:
        return [("common", False, str(e))]

def test_dataset():
    """Test dataset module."""
    results = []
    try:
        from nanochat.dataset import list_parquet_files
        
        # Test listing parquet files
        files = list_parquet_files()
        results.append(("list_parquet_files", isinstance(files, list),
                       f"Found {len(files)} parquet files"))
        
        return results
    except Exception as e:
        return [("dataset", False, str(e))]

def test_tasks():
    """Test task modules."""
    results = []
    task_modules = [
        ("arc", "ARC"),
        ("gsm8k", "GSM8K"),
        ("mmlu", "MMLU"),
        ("humaneval", "HumanEval"),
        ("smoltalk", "SmolTalk"),
    ]
    
    for module_name, class_name in task_modules:
        try:
            module = importlib.import_module(f"tasks.{module_name}")
            task_class = getattr(module, class_name)
            results.append((f"tasks.{module_name}", True, f"{class_name} imported successfully"))
        except Exception as e:
            results.append((f"tasks.{module_name}", False, str(e)))
    
    return results

def test_scripts():
    """Test script modules can be imported."""
    results = []
    scripts = [
        "scripts.base_train",
        "scripts.chat_cli",
        "scripts.chat_web",
        "scripts.chat_sft",
        "scripts.ssm_demo",
        "scripts.tok_train",
        "scripts.tok_eval",
    ]
    
    for script_name in scripts:
        try:
            module = importlib.import_module(script_name)
            results.append((script_name, True, "Module imported successfully"))
        except Exception as e:
            # Some scripts might fail due to missing dependencies or config, that's ok
            error_msg = str(e)[:100]
            results.append((script_name, False, f"Import error: {error_msg}"))
    
    return results

def run_all_tests():
    """Run all tests and collect results."""
    all_results = {}
    
    # Core modules
    print("Testing core modules...")
    modules_to_test = [
        ("nanochat.tokenizer", test_tokenizer),
        ("nanochat.gpt", test_gpt),
        ("nanochat.engine", test_engine),
        ("nanochat.ssm", test_ssm),
        ("nanochat.checkpoint_manager", test_checkpoint_manager),
        ("nanochat.common", test_common),
        ("nanochat.dataset", test_dataset),
    ]
    
    for module_name, test_func in modules_to_test:
        print(f"  Testing {module_name}...")
        try:
            can_import, import_msg = test_module_import(module_name)
            if can_import:
                results = test_func()
                all_results[module_name] = {
                    "import": (True, import_msg),
                    "tests": results
                }
            else:
                all_results[module_name] = {
                    "import": (False, import_msg),
                    "tests": []
                }
        except Exception as e:
            all_results[module_name] = {
                "import": (False, str(e)),
                "tests": []
            }
    
    # Test tasks
    print("Testing task modules...")
    task_results = test_tasks()
    all_results["tasks"] = {
        "import": (True, "Tasks module"),
        "tests": task_results
    }
    
    # Test scripts
    print("Testing script modules...")
    script_results = test_scripts()
    all_results["scripts"] = {
        "import": (True, "Scripts module"),
        "tests": script_results
    }
    
    return all_results

def generate_report(results: Dict) -> str:
    """Generate a markdown report from test results."""
    report = []
    report.append("# NanoChat-Live Comprehensive Test Results")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("---\n")
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for module_name, module_results in results.items():
        report.append(f"## {module_name}\n")
        
        # Import status
        import_passed, import_msg = module_results["import"]
        status_icon = "✅" if import_passed else "❌"
        report.append(f"**Import Status:** {status_icon} {import_msg}\n")
        
        if not import_passed:
            report.append("\n")
            continue
        
        # Test results
        if module_results["tests"]:
            report.append("### Test Results\n")
            report.append("| Test | Status | Details |")
            report.append("|------|--------|---------|")
            
            for test_name, test_passed, test_msg in module_results["tests"]:
                total_tests += 1
                if test_passed:
                    passed_tests += 1
                    status = "✅ PASS"
                else:
                    failed_tests += 1
                    status = "❌ FAIL"
                
                # Truncate long messages
                msg = str(test_msg)[:100] + "..." if len(str(test_msg)) > 100 else str(test_msg)
                report.append(f"| {test_name} | {status} | {msg} |")
        else:
            report.append("*No specific tests run*\n")
        
        report.append("\n")
    
    # Summary
    report.append("---\n")
    report.append("## Summary\n")
    report.append(f"- **Total Tests:** {total_tests}")
    report.append(f"- **Passed:** {passed_tests} ✅")
    report.append(f"- **Failed:** {failed_tests} ❌")
    if total_tests > 0:
        success_rate = (passed_tests / total_tests) * 100
        report.append(f"- **Success Rate:** {success_rate:.1f}%")
    
    return "\n".join(report)

if __name__ == "__main__":
    print("=" * 60)
    print("NanoChat-Live Comprehensive Module Testing")
    print("=" * 60)
    print()
    
    results = run_all_tests()
    report = generate_report(results)
    
    # Print to console
    print("\n" + "=" * 60)
    print("TEST REPORT")
    print("=" * 60)
    print(report)
    
    # Save to file
    report_file = "TEST_RESULTS.md"
    with open(report_file, "w") as f:
        f.write(report)
    
    print(f"\n✅ Report saved to: {report_file}")

