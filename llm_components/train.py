import torch
from model import GPT

data_file = "data.txt"

with open(data_file, "r") as f:
    data = f.read()

print(data[:50])
