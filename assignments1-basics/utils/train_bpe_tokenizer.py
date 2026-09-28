import regex as re
from collections import Counter

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    word_counts = Counter(match.group() for match in re.finditer(PAT, text))
    
    word_freqs = {}
    for word, count in word_counts.items():
        word_bytes_tuple = tuple(list(word.encode('utf-8')))
        word_freqs[word_bytes_tuple] = count
    
    vocab = {i: bytes([i]) for i in range(256)}

    for token in special_tokens:
        new_id = len(vocab)
        vocab[new_id] = token.encode('utf-8')

    merges = []
    
    while len(vocab) < vocab_size:
        stats = {}
        
        for word, count in word_freqs.items():
            for x, y in zip(word[:-1], word[1:]):
                stats[(x, y)] = stats.get((x, y), 0) + count
        
        if not stats:
            break
                    
        best_pair = max(stats, key=lambda p: (stats[p], p))
        
        new_id = len(vocab)
        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

        bytes_first = vocab[best_pair[0]]
        bytes_second = vocab[best_pair[1]]
        merges.append((bytes_first, bytes_second))

        next_word_freqs = {}
        for word, count in word_freqs.items():
            new_word = []
            i = 0 
            while i < len(word):
                if i + 1 < len(word) and word[i] == best_pair[0] and word[i+1] == best_pair[1]:
                    new_word.append(new_id)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            next_word_freqs[tuple(new_word)] = count

        word_freqs = next_word_freqs

    return vocab, merges