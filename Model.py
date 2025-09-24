import torch
import torch.nn as nn
import torch.optim as optim
from Embedding import InputEmbedding, OutputUnembedding
from Tokenizer import BytePairEncoding
from Transformer import CausalDecoderOnlyTransformer
import pickle
from Attention import apply_rope
from tqdm import tqdm
from Dataloader import TextCorpusDataset, collate_fn
from torch.utils.data import DataLoader, Dataset

class Model(nn.Module):
    def __init__(self, num_layers = 4, embedding_size = 256, num_heads = 8, mlp_ratio = 4, bias = False, vocab_size = 1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.bias = bias
        self.input_embedding = InputEmbedding(vocab_size = self.vocab_size, output_dim = embedding_size)
        self.decoder_transformer = CausalDecoderOnlyTransformer(num_layers = num_layers, embedding_size = embedding_size, num_heads = num_heads, mlp_ratio = mlp_ratio, bias = bias)
        self.output_unembedding = OutputUnembedding(self.input_embedding)
    
    def forward(self, input_ids):
        input_ids = self.input_embedding(input_ids)
        input_ids = apply_rope(input_ids)
        input_ids = self.decoder_transformer(input_ids)
        input_ids = self.output_unembedding(input_ids)
        return input_ids

if __name__ == "__main__":
    device = "cuda"
    model = Model().to(device)
    #x = "Hello my name is Anthony Gonzalves"
    #input_ids = model.tokenize(x)
    #output = model(input_ids)
    epochs = 1000
    block_size = 256
    batch_size = 128
    tokenizer_path = '/workspace/Personal/Transformer/tokenizer_smaller.pkl'
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    dataset = TextCorpusDataset(txt_dir = "/workspace/Personal/Transformer/shakespeare-dataset/text", tokenizer = tokenizer, block_size = block_size)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True, collate_fn = collate_fn)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    for epoch in tqdm(range(epochs)):
        losses = 0.0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            logits = outputs.permute(0,2,1)
            loss = criterion(logits, targets)
            losses += loss.item()
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            #print("Done for batch", batch_idx)
        print("avg losses", losses/(batch_idx+1))
        torch.save(model.state_dict(), f'model_epoch_{epoch}.pt')

