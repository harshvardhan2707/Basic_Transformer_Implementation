import os
from torch.utils.data import DataLoader, Dataset
import torch
import pickle

class TextCorpusDataset(Dataset):
    def __init__(self, txt_dir, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.doc_tokens_list = []
        self.idx_map = []

        for fname in os.listdir(txt_dir):
            if not fname.endswith('.txt'):
                continue
            path = os.path.join(txt_dir, fname)
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            tokens = self.tokenizer.encoding(text)
            self.doc_tokens_list.append(tokens)
            print("Done path", path)

        for doc_idx,tokens in enumerate(self.doc_tokens_list):
            n = len(tokens)
            if(n <= block_size):#Currently just dropping ones which are of lesser size than 
                continue
            for token_idx in range(0, (n-1)//block_size):
                self.idx_map.append((doc_idx, token_idx*block_size))

    def __len__(self):
        return len(self.idx_map)

    def __getitem__(self, idx):
        doc_idx, start_idx = self.idx_map[idx]
        input_ids = self.doc_tokens_list[doc_idx][start_idx:start_idx + self.block_size]
        target_ids = self.doc_tokens_list[doc_idx][start_idx + 1:start_idx + 1 + self.block_size]
        return {'input_ids': torch.tensor(input_ids, dtype = torch.long),
                'target_ids': torch.tensor(target_ids, dtype = torch.long)
                }

def collate_fn(batch):
    inputs = torch.stack([b['input_ids'] for b in batch], dim=0)
    targets = torch.stack([b['target_ids'] for b in batch], dim=0)
    return inputs, targets

if __name__ == "__main__":
    tokenizer_path = '/home/cv-research/Transformer/tokenizer_smaller.pkl'
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    block_size = 256
    batch_size = 32

    dataset = TextCorpusDataset(txt_dir = "/home/cv-research/Transformer/shakespeare-dataset/text", tokenizer = tokenizer, block_size = block_size)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True, collate_fn = collate_fn)
    for batch in dataloader:
        breakpoint()
    X = next(dataloader)
