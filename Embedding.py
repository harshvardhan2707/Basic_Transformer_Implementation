import torch
import torch.nn as nn
from Tokenizer import BytePairEncoding
import pickle

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size = 1024, output_dim = 256):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = output_dim, max_norm = 1.0)

    def forward(self, token_ids):
        embeddings = self.embedding(token_ids)
        return embeddings

class OutputUnembedding(nn.Module):
    def __init__(self, input_embedding):
        self.output_embedding = input_embedding.embedding

    def forward(self, x):
        x = x@(self.output_embedding.T)
        return x

if __name__ == "__main__":
    with open('tokenizer_smaller.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    word_embeddings = InputEmbedding(vocab_size = tokenizer.vocab_size, output_dim = 256)
    String = "My name is Anthony Gonzalves"
    tokens = torch.tensor(tokenizer.encoding(String))
    F = word_embeddings(tokens)
    breakpoint()
