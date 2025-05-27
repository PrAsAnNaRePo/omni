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

    def encode(self, text: str):
        substrings = self.token_pattern.findall(text)
        tokens = []
        for chunk in substrings:
            tokens.extend(chunk.encode('utf-8'))
        return self.__apply_bpe_merges(tokens, self.merges)

    def decode(self, tokens: list[int]):
        return ''.join(self.vocab[i].decode('utf-8') for i in tokens)

# usage
#tok = Tokenizer()
#t = tok.encode("return [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]")
#print(len(t))
#print(tok.decode(t))
