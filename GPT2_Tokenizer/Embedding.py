import torch
import torch.nn as nn
import pickle

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size = 1024, output_dim = 256):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = output_dim, max_norm = 1.0)

    def forward(self, token_ids):
        embeddings = self.embedding(token_ids) #(batch_size, seq_length) --> (batch_size, seq_length, output_dim)
        return embeddings

class OutputUnembedding(nn.Module):
    def __init__(self, input_embedding):
        super().__init__()
        self.output_embedding = input_embedding.embedding.weight

    def forward(self, x):
        x = x@(self.output_embedding.T) #(batch_size, seq_length, output_dim) -->  (batch_size, seq_length, vocab_size)
        return x

if __name__ == "__main__":
    print("Embedding")
