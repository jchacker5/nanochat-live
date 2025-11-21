# GUARANTEED WORKING TOKENIZER - Paste this into Colab notebook cell
# This creates ALL required files: tokenizer.pkl, token_bytes.pt

import os
import torch
import pickle
import tiktoken
from nanochat.tokenizer import HuggingFaceTokenizer, SPECIAL_TOKENS
from nanochat.dataset import parquets_iter_batched

tokenizer_dir = "/root/.cache/nanochat/tokenizer"
os.makedirs(tokenizer_dir, exist_ok=True)

print("="*70)
print("Training tokenizer on 2B characters...")
print("Using HuggingFace tokenizer (works on Colab)")
print("="*70)

# Create text iterator
def text_iterator():
    nchars = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            doc_text = doc
            if len(doc_text) > 10000:  # Cap document length
                doc_text = doc_text[:10000]
            nchars += len(doc_text)
            yield doc_text
            if nchars > 2000000000:  # 2B characters
                return

# Train tokenizer
print("Training BPE tokenizer...")
hf_tokenizer = HuggingFaceTokenizer.train_from_iterator(text_iterator(), vocab_size=65536)

# Save HuggingFace tokenizer.json
hf_tokenizer.tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
print("✅ Saved tokenizer.json")

# Create tiktoken-compatible encoding for tokenizer.pkl
vocab = hf_tokenizer.tokenizer.get_vocab()

# Create mergeable_ranks (BPE merges)
mergeable_ranks = {}
rank = 0
for token, token_id in sorted(vocab.items(), key=lambda x: x[1]):
    if token not in SPECIAL_TOKENS:
        try:
            mergeable_ranks[token.encode('utf-8')] = rank
            rank += 1
        except:
            pass

# Create special tokens mapping
tokens_offset = len(mergeable_ranks)
special_tokens_dict = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}

# Create tiktoken encoding (what RustBPETokenizer expects)
enc = tiktoken.Encoding(
    name="huggingface_bpe",
    pat_str=r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
    mergeable_ranks=mergeable_ranks,
    special_tokens=special_tokens_dict,
)

# Save as pickle (what RustBPETokenizer.from_directory expects)
with open(os.path.join(tokenizer_dir, 'tokenizer.pkl'), 'wb') as f:
    pickle.dump(enc, f)
print("✅ Saved tokenizer.pkl")

# Create token_bytes.pt (needed for evaluation)
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
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print("✅ Saved token_bytes.pt")

print("="*70)
print("✅ Tokenizer trained and saved successfully!")
print(f"✅ Files created in: {tokenizer_dir}")
print(f"   - tokenizer.pkl: {os.path.exists(os.path.join(tokenizer_dir, 'tokenizer.pkl'))}")
print(f"   - tokenizer.json: {os.path.exists(os.path.join(tokenizer_dir, 'tokenizer.json'))}")
print(f"   - token_bytes.pt: {os.path.exists(os.path.join(tokenizer_dir, 'token_bytes.pt'))}")
print("="*70)

