
import re

class CustomTokenizer():
    def __init__(self, vocab_size=5000):
        assert vocab_size > 256, "Vocab size should be atleast 256"
        self.vocab_size = vocab_size
        self.vocab =  {idx: bytes([idx]) for idx in range(256)}
        self.merges = None
    
    def get_compression_ratio(self, text):
        return 
    
    def get_words_from_sentence(self, text):
        mo = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        tokens = mo.findall(text)
        return tokens
    
    def train_tokenizer(self, text):
        self.merges = {} # (int, int) -> int
        num_merges = self.vocab_size - 256

        words = self.get_words_from_sentence(text)
        for i in words:
            ids = self.encode(i)

            for i in range(num_merges):
                stats = self.get_stats(ids)
                pair = max(stats, key=stats.get)
                idx = 256 + i
                print(f"merging {pair} into a new token {idx}")
                ids = self.merge(ids, pair, idx)
                self.merges[pair] = idx
        
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]


    def merge(self, ids, pair, idx):
        # in the list of ints (ids), replace all consecutive occurences of pair with the new token idx
        new_ids = []
        i = 0
        while i < len(ids):
            # if we are not at the very last position and the pair matches, replace it
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text
    
    def get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]): # Pythonic way to iterate consecutive elements
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def encode(self, text):
        # given a string, return list of integers (the tokens)
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break # nothing else can be merged
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens