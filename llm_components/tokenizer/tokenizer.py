import regex
from collections import Counter
from copy import copy

class Tokenizer:
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}
        self.merges = {}
        # compile once
        pat_str = r"|".join([
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""\p{N}{1,3}""",
            r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
            r"""\s*[\r\n]+""",
            r"""\s+(?!\S)""",
            r"""\s+""",
        ])
        self.token_pattern = regex.compile(pat_str)

    def __get_pairs(self, tokens: list[int]) -> list[tuple[int,int]]:
        return [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]

    def __get_pairs_count(self, pairs: list[tuple[int,int]]) -> dict:
        return Counter(pairs)

    def __get_merges(self, pairs_count: dict, max_tok: int) -> dict:
        merges = {}
        for pair, count in sorted(pairs_count.items(), key=lambda x: x[1], reverse=True):
            if count > 1:
                max_tok += 1
                merges[pair] = max_tok
        return merges

    def __apply_bpe_merges(self, raw_tokens, merges):
        merged = copy(raw_tokens)
        for pair, new_tok in merges.items():
            idx = 0
            while idx < len(merged)-1:
                if (merged[idx], merged[idx+1]) == pair:
                    merged[idx] = new_tok
                    merged.pop(idx+1)
                else:
                    idx += 1
        return merged

    def init_vocab(self, data: str):
        self.vocab = {i: bytes([i]) for i in range(256)}
        current_token = 255

        substrings = self.token_pattern.findall(data)
        tokens = []
        for chunk in substrings:
            tokens.extend(chunk.encode('utf-8'))
        tokens = tokens[:10000]  # for demo

        while True:
            pairs = self.__get_pairs(tokens)
            pairs_count = self.__get_pairs_count(pairs)
            if pairs_count and max(pairs_count.values()) >= 2:
                print('Merging...')
                self.merges.update(self.__get_merges(pairs_count, current_token))
                tokens = self.__apply_bpe_merges(tokens, self.merges)
                for (p0,p1), new_tok in self.merges.items():
                    self.vocab[new_tok] = self.vocab[p0] + self.vocab[p1]
                current_token = max(tokens)
                print("vocab size: ", len(self.vocab))
            else:
                break

        self.reverse_vocab = {v:k for k,v in self.vocab.items()}

    def encode(self, text: str):
        substrings = self.token_pattern.findall(text)
        tokens = []
        for chunk in substrings:
            tokens.extend(chunk.encode('utf-8'))
        return self.__apply_bpe_merges(tokens, self.merges)

    def decode(self, tokens: list[int]):
        return ''.join(self.vocab[i].decode('utf-8') for i in tokens)

# Usage:
tok = Tokenizer()
tok.init_vocab(open('data.txt').read())
t = tok.encode("return [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]")
print(len(t))
print(tok.decode(t))
