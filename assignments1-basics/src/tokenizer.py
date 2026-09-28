import pickle
import regex as re
from collections.abc import Iterable, Iterator

class Tokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        self.bytes_to_id = {v: k for k, v in vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}

        next_id = max(vocab.keys()) + 1 if vocab else 0

        if special_tokens:
            for token in special_tokens:
                bytes_token = token.encode("utf-8")
                if bytes_token not in self.bytes_to_id:
                    self.vocab[next_id] = bytes_token
                    self.bytes_to_id[bytes_token] = next_id
                    next_id += 1

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        if not text:
            return []

        PAT = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        words = PAT.findall(text)

        final_ids = []

        for word in words:
            tokens = [bytes([b]) for b in word.encode("utf-8")]

            while len(tokens) >= 2:
                pairs = list(zip(tokens[:-1], tokens[1:]))

                best_pair = min(
                    pairs, key=lambda p: self.merge_ranks.get(p, float("inf"))
                )

                if best_pair not in self.merge_ranks:
                    break

                new_tokens = []
                i = 0
                while i < len(tokens):
                    if (
                        i < len(tokens) - 1
                        and (tokens[i], tokens[i + 1]) == best_pair
                    ):
                        new_tokens.append(tokens[i] + tokens[i + 1])
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens

            for token in tokens:
                final_ids.append(self.bytes_to_id[token])

        return final_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            token_ids = self.encode(chunk)

            for token_id in token_ids:
                yield token_id


    def decode(self, ids: list[int]) -> str:
        byte_chunks = []
        for i, byte in enumerate(ids):
            byte_chunks.append(self.vocab[byte])

        return (b"".join(byte_chunks)).decode('utf-8', errors="replace")