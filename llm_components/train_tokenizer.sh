#!/bin/bash

cd tokenizer && uv run train.py --data "../data.txt" --special_tokens "<eos>" "<pad>" "<|im_start|>" "<|im_end|>"
