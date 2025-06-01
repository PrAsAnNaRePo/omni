import pickle
import regex
from collections import Counter
from copy import copy

class Tokenizer:
    def __init__(self, vocab_path="vocab.bpe", merges_path="merges.bpe"):
        with open(vocab_path, 'rb') as vocab_file:
            self.vocab = pickle.load(vocab_file)

        with open(merges_path, 'rb') as merges_file:
            self.merges = pickle.load(merges_file)

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

    def __apply_merge(self, tokens: list[int], pair: tuple[int, int], new_token: int) -> list[int]:
        merged = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                merged.append(new_token)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    def encode(self, text: str) -> list[int]:
        tokens = list(text.encode('utf-8'))
        
        for pair, new_token_id in self.merges.items():
            tokens = self.__apply_merge(tokens, pair, new_token_id)
        
        return tokens

    def decode(self, tokens: list[int]) -> str:
        byte_data = b''
        for token_id in tokens:
            if token_id in self.vocab:
                byte_data += self.vocab[token_id]
            else:
                byte_data += self.vocab.get(256, b'<|UNK|>')
        
        try:
            return byte_data.decode('utf-8')
        except UnicodeDecodeError:
            return byte_data.decode('utf-8', errors='replace')

# usage
# tok = Tokenizer()
# t = tok.encode("return [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]")
# print(len(t))
# print(tok.decode(t))
