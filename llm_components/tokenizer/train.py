import argparse
import time
import regex
from collections import Counter
from copy import copy
import pickle
from tqdm import tqdm

class TrainTokenizer:
    def __init__(self, special_tokens: list, vocab_save_file: str, merges_save_file: str):
        self.vocab = {}
        self.merges = {}
        self.special_tokens = special_tokens or []
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

    def __get_pairs(self, tokens: list[int]) -> list[tuple[int, int]]:
        return [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]

    def __get_pairs_count(self, pairs: list[tuple[int, int]]) -> dict:
        return Counter(pairs)

    def __apply_merge(self, tokens: list[int], pair: tuple[int, int], new_token: int) -> list[int]:
        """Apply a single merge operation to the token list"""
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

    def train(self, data: str, max_vocab_size: int = 10000, logging_step=100):
        self.vocab = {i: bytes([i]) for i in range(256)}
        
        self.vocab[256] = b'<|UNK|>'
        current_token_id = 256
        
        substrings = self.token_pattern.findall(data)
        tokens = []
        for chunk in substrings:
            tokens.extend(chunk.encode('utf-8'))
        
        # tokens = tokens[:100_000]
        print(f"Initial tokens: {len(tokens)}")
        print(f"Initial vocab size: {len(self.vocab)}")
        
        iteration = 0
        p_bar = tqdm(total=max_vocab_size - len(self.vocab))

        while len(self.vocab) < max_vocab_size:
            old_vocab_len = len(self.vocab)
            pairs = self.__get_pairs(tokens)
            if not pairs:
                break
            
            pairs_count = self.__get_pairs_count(pairs)
            
            if not pairs_count:
                break
            
            most_frequent_pair = max(pairs_count.items(), key=lambda x: x[1])
            pair, count = most_frequent_pair
            
            if count < 2:
                break
            
            current_token_id += 1
            new_token_id = current_token_id
            
            self.merges[pair] = new_token_id
            
            self.vocab[new_token_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            
            tokens = self.__apply_merge(tokens, pair, new_token_id)
            
            p_bar.update(len(self.vocab) - old_vocab_len)
            iteration += 1
            if iteration % logging_step == 0:
                print(f"Iteration {iteration}: Merged {pair} -> {new_token_id} (count: {count})")
                print(f"Vocab size: {len(self.vocab)}, Tokens: {len(tokens)}")
        
        if self.special_tokens:
            for token in self.special_tokens:
                current_token_id += 1
                self.vocab[current_token_id] = token.encode('utf-8') if isinstance(token, str) else token
        
        print(f"Training completed!")
        print(f"Final vocab size: {len(self.vocab)}")
        print(f"Number of merges: {len(self.merges)}")
        
        with open(self.vocab_save_file, 'wb') as vocab_file:
            pickle.dump(self.vocab, vocab_file)
        
        with open(self.merges_save_file, 'wb') as merge_file:
            pickle.dump(self.merges, merge_file)
        
        print(f"Saved vocab to {self.vocab_save_file}")
        print(f"Saved merges to {self.merges_save_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to training data file")
    parser.add_argument("--vocab_file_name", default="vocab.bpe", help="Vocabulary save file")
    parser.add_argument("--merge_file_name", default="merges.bpe", help="Merges save file")
    parser.add_argument("--special_tokens", nargs="*", help="Special tokens to add")
    parser.add_argument("--max_vocab_size", type=int, default=10000, help="Maximum vocabulary size")
    parser.add_argument("--logging_step", type=int, default=100, help="Logging step size")
    args = parser.parse_args()

    try:
        with open(args.data, 'r', encoding='utf-8') as f:
            data = f.read()
    except FileNotFoundError:
        print(f"Error: Could not find data file '{args.data}'")
        exit(1)
    except UnicodeDecodeError:
        with open(args.data, 'r', encoding='utf-8', errors='replace') as f:
            data = f.read()
    
    tokenizer = TrainTokenizer(
        special_tokens=args.special_tokens,
        vocab_save_file=args.vocab_file_name,
        merges_save_file=args.merge_file_name
    )
    
    start_time = time.time()
    tokenizer.train(data=data, max_vocab_size=args.max_vocab_size, logging_step=args.logging_step)
    print(f"All done in {(time.time() - start_time):.2f}s")