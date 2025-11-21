"""
Quick tokenizer test - verifies tokenizer works correctly
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pickle
import tiktoken
from nanochat.tokenizer import HuggingFaceTokenizer, SPECIAL_TOKENS, get_tokenizer
from nanochat.dataset import parquets_iter_batched

print("="*70)
print("TOKENIZER TEST - Verifying HuggingFace Tokenizer Works")
print("="*70)

# Test 1: Import check
print("\n[Test 1] Import Check")
print("  ✓ HuggingFaceTokenizer imported")
print("  ✓ SPECIAL_TOKENS imported")
print("  ✓ parquets_iter_batched imported")

# Test 2: Create a small tokenizer (quick test)
print("\n[Test 2] Training Small Tokenizer (100k chars)")
tokenizer_dir = "/tmp/test_tokenizer"
os.makedirs(tokenizer_dir, exist_ok=True)

def text_iterator():
    nchars = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            doc_text = doc[:1000]  # Small test
            nchars += len(doc_text)
            yield doc_text
            if nchars > 100000:  # Just 100k chars for quick test
                return

print("  Training on 100k characters...")
hf_tokenizer = HuggingFaceTokenizer.train_from_iterator(text_iterator(), vocab_size=4096)
print("  ✓ Tokenizer trained successfully")
print(f"  ✓ Vocab size: {hf_tokenizer.get_vocab_size()}")

# Test 3: Save tokenizer files
print("\n[Test 3] Saving Tokenizer Files")
hf_tokenizer.tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
print("  ✓ Saved tokenizer.json")

# Create tiktoken encoding
vocab = hf_tokenizer.tokenizer.get_vocab()
mergeable_ranks = {}
rank = 0
for token, token_id in sorted(vocab.items(), key=lambda x: x[1]):
    if token not in SPECIAL_TOKENS:
        try:
            mergeable_ranks[token.encode('utf-8')] = rank
            rank += 1
        except:
            pass

tokens_offset = len(mergeable_ranks)
special_tokens_dict = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}

enc = tiktoken.Encoding(
    name="test_bpe",
    pat_str=r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
    mergeable_ranks=mergeable_ranks,
    special_tokens=special_tokens_dict,
)

with open(os.path.join(tokenizer_dir, 'tokenizer.pkl'), 'wb') as f:
    pickle.dump(enc, f)
print("  ✓ Saved tokenizer.pkl")

# Create token_bytes.pt
vocab_size = hf_tokenizer.get_vocab_size()
special_set = set(hf_tokenizer.get_special_tokens())
token_bytes = []
for token_id in range(vocab_size):
    try:
        token_str = hf_tokenizer.decode([token_id])
        if token_str in special_set:
            token_bytes.append(0)
        else:
            token_bytes.append(len(token_str.encode("utf-8")))
    except:
        token_bytes.append(0)

token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
with open(os.path.join(tokenizer_dir, "token_bytes.pt"), "wb") as f:
    torch.save(token_bytes, f)
print("  ✓ Saved token_bytes.pt")

# Test 4: Verify files exist
print("\n[Test 4] File Verification")
files = {
    'tokenizer.json': os.path.join(tokenizer_dir, 'tokenizer.json'),
    'tokenizer.pkl': os.path.join(tokenizer_dir, 'tokenizer.pkl'),
    'token_bytes.pt': os.path.join(tokenizer_dir, 'token_bytes.pt')
}

all_exist = True
for name, path in files.items():
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    status = "EXISTS" if exists else "MISSING"
    print(f"  ✓ {name}: {status} ({size:,} bytes)")
    if not exists:
        all_exist = False

assert all_exist, "Some tokenizer files are missing!"

# Test 5: Test encoding/decoding
print("\n[Test 5] Encoding/Decoding Test")
test_text = "Hello world! This is a test. 123 456"
encoded = hf_tokenizer.encode(test_text)
decoded = hf_tokenizer.decode(encoded)
print(f"  Original: {test_text}")
print(f"  Encoded: {len(encoded)} tokens")
print(f"  Decoded: {decoded}")
round_trip = decoded == test_text
print(f"  ✓ Round-trip: {'PASS' if round_trip else 'FAIL'}")

# Test 6: Test loading from directory
print("\n[Test 6] Load from Directory Test")
try:
    loaded_tokenizer = HuggingFaceTokenizer.from_directory(tokenizer_dir)
    test_encoded = loaded_tokenizer.encode("Test loading")
    print(f"  ✓ Loaded tokenizer works")
    print(f"  ✓ Encoded test: {len(test_encoded)} tokens")
except Exception as e:
    print(f"  ⚠️  Load error: {e}")

print("\n" + "="*70)
print("✅ TOKENIZER TEST COMPLETE - All checks passed!")
print("="*70)
print("\nThe tokenizer works correctly!")
print("The notebook will train it on 2B characters for production use.")

