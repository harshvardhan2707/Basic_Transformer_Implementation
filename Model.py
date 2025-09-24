import torch
import torch.nn as nn
from Embedding import InputEmbedding, OutputUnembedding
from Tokenizer import BytePairEncoding
from Transformer import CausalDecoderOnlyTransformer
import pickle
from Attention import apply_rope
import tqdm

class Model(nn.Module):
    def __init__(self, tokenizer_path, num_layers = 4, embedding_size = 256, num_heads = 8, mlp_ratio = 4, bias = False):
        super().__init__()
        self.vocab_size = self.tokenizer.vocab_size
        self.embedding_dim = embedding_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.bias = bias
        self.input_embedding = InputEmbedding(vocab_size = self.vocab_size, output_dim = embedding_size)
        self.decoder_transformer = CausalDecoderOnlyTransformer(num_layers = num_layers, embedding_size = embedding_size, num_heads = num_heads, mlp_ratio = mlp_ratio, bias = bias)
        self.output_unembedding = OutputUnembedding(self.input_embedding)
    
    def forward(self, input_ids):
        breakpoint()
        input_ids = self.input_embedding(torch.unsqueeze(input_ids, dim = 0))
        input_ids = apply_rope(input_ids)
        input_ids = self.decoder_transformer(input_ids)
        input_ids = self.output_unembedding(input_ids)
        return input_ids

if __name__ == "__main__":
    model = Model('/home/cv-research/Transformer/tokenizer_smaller.pkl')
    #x = "Hello my name is Anthony Gonzalves"
    #input_ids = model.tokenize(x)
    #output = model(input_ids)
    epochs = 1000
    block_size = 256
    batch_size = 32
    tokenizer_path = '/home/cv-research/Transformer/tokenizer_smaller.pkl'
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    dataset = TextCorpusDataset(txt_dir = "/home/cv-research/Transformer/shakespeare-dataset/text", tokenizer = tokenizer, block_size = block_size)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True, collate_fn = collate_fn)
    criterion = nn.CrossEntropyLoss()
    for epoch in tqdm(range(epochs)):
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

