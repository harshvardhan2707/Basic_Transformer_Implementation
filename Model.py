import os
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
import wandb
import argparse

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
    parser = argparse.ArgumentParser(description="Model for training a very simple decoder only transformer")
    parser.add_argument("--project_name",type=str, required=True,  help="Name of project(this will also be the saving directory")
    parser.add_argument("--optimizer",type=str, required=True,  help="Which optimizer to use")
    parser.add_argument("--learning_rate",type=float, required=True,  help="The learning rate to use")
    parser.add_argument("--epochs", type=int, required=True, help="Total number of epochs")
    parser.add_argument("--save_freq", type=int, default=10, help="Save frequency of model")
    parser.add_argument("--block_size", type=int, default=256, help="Block size, basically till what token number the predictions will it be good")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    args = parser.parse_args()
    wandb.login()
    run = wandb.init(
            project=args.project_name,
            config={
                'optimizer': args.optimizer,
                "lr": args.learning_rate,
                "epochs": args.epochs,
                "save_freq": args.save_freq,
                "block_size": args.block_size,
                "batch_size": args.batch_size
                })
    device = "cuda"
    model = Model().to(device)
    #x = "Hello my name is Anthony Gonzalves"
    #input_ids = model.tokenize(x)
    #output = model(input_ids)
    epochs = args.epochs
    block_size = args.block_size
    batch_size = args.batch_size
    tokenizer_path = '/workspace/Personal/Transformer/tokenizer_smaller.pkl'
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    dataset = TextCorpusDataset(txt_dir = "/workspace/Personal/Transformer/shakespeare-dataset/text", tokenizer = tokenizer, block_size = block_size)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True, collate_fn = collate_fn)
    criterion = nn.CrossEntropyLoss()
    if(args.optimizer.lower() == "adamw"):
        optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,            # better than 1e-3 for small model
        betas=(0.9, 0.95),  # GPT-2 defaults
        eps=1e-8    # turn off for Shakespeare
        )
    else:
        raise ValueError(f"Not implemented {args.optimizer} yet")
    Min_loss = 1e9
    loss_list = []
    BASE_PATH = 'runs/' + args.project_name
    os.makedirs(BASE_PATH, exist_ok=True)

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
        loss_list += [losses]
        wandb.log({"loss":losses})
        print(sum(loss_list)/len(loss_list))
        if(epoch%args.save_freq == 0):
            torch.save(model.state_dict(), f'{BASE_PATH}/model_epoch_{epoch}.pt')
        if(losses<Min_loss):
            torch.save(model.state_dict(), f'{BASE_PATH}/best_model.pt')
            Min_loss = losses
    run.finish()
