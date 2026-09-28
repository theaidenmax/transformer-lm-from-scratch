import numpy as np
from src.tokenizer import Tokenizer

VOCAB_FILEPATH = "../datasets/tokenizer_vocab.pkl"
MERGES_FILEPATH = "../datasets/tokenizer_merges.pkl"
SPECIAL_TOKENS = ["<|endoftext|>"]

CORPUS = "../datasets/owt_small.txt"

tokenizer = Tokenizer.from_files(
    vocab_filepath=VOCAB_FILEPATH,
    merges_filepath=MERGES_FILEPATH,
    special_tokens=SPECIAL_TOKENS,
)

with open(CORPUS, "r", encoding="utf-8") as f:
    text = f.read()

tokens = tokenizer.encode(text)

tokens_np = np.array(tokens, dtype=np.uint16)

np.save("../datasets/owt_train.npy", tokens_np)

print(f"Tokenization complete. Total tokens: {len(tokens_np)}")