import argparse
import regex
from collections import Counter
from copy import copy
import pickle

class TrainTokenizer:
    def __init__(self, special_tokens:list, vocab_save_file: str, merges_save_file: str):
        self.vocab = {}
        self.merges = {}
        self.special_tokens = special_tokens
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

        self.vocab_save_file = vocab_save_file
        self.merges_save_file = merges_save_file

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

    def train(self, data: str):
        self.vocab = {i: bytes([i]) for i in range(256)}

        # adding +1 for UNK token
        self.vocab[256] = '<|UNK|>'
        current_token = 255 + 1

        substrings = self.token_pattern.findall(data)
        tokens = []
        for chunk in substrings:
            tokens.extend(chunk.encode('utf-8'))
        tokens = tokens[:10000]

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
        
        curr_max_vocab_size = max(list(self.vocab.keys()))
        if self.special_tokens:
            for token in self.special_tokens:
                curr_max_vocab_size+=1
                self.vocab[curr_max_vocab_size] = token

        with open(self.vocab_save_file, 'wb') as vocab_file:
            pickle.dump(self.vocab, vocab_file)
        
        with open(self.merges_save_file, 'wb') as merge_file:
            pickle.dump(self.merges, merge_file)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--vocab_file_name", default="vocab.bpe")
    parser.add_argument("--merge_file_name", default="merges.bpe")
    parser.add_argument("--special_tokens", nargs="*", help="Special tokens")
    args = parser.parse_args()

    data = open(args.data).read()
    print("data preview ===>")
    print(data[:50])
    print("====>")
    tok = TrainTokenizer(
        special_tokens=args.special_tokens,
        vocab_save_file=args.vocab_file_name,
        merges_save_file=args.merge_file_name
    )
    print("Initiating training...")
    tok.train(data=data)
