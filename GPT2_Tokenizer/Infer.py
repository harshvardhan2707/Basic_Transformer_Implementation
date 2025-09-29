from Model import Model
import torch
import numpy as np
import argparse
import pickle
from transformers import AutoTokenizer

def generate_next_token(probs, np_array):
    return np.random.choice(np_array, p = probs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Gibbrish")
    parser.add_argument("--model_path", type = str, required=True, help='Provide Model Path')
    parser.add_argument("--model_input", type = str, required=True, help='Provide Input to the Model')
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(f"{input('Enter tokenizer path/huggingface link: ')}")
    model = Model(vocab_size = len(tokenizer.get_vocab())).to('cuda')
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    token_ids = tokenizer(args.model_input, return_tensors="pt")['input_ids'].squeeze()
    tot_length = 256
    array = np.arange(len(tokenizer.get_vocab()))
    print(tokenizer.decode(token_ids), end = "", flush=True)
    while(len(token_ids)< tot_length):
        outputs = model(torch.unsqueeze(token_ids, dim = 0).to('cuda'))
        logits = outputs[0, -1]
        probs = torch.softmax(logits, dim = -1)
        token_ids = torch.cat((token_ids, torch.tensor([generate_next_token(probs.cpu().detach().numpy(), array)])), dim = 0)
        new_token = token_ids[-1].item()
        print(tokenizer.decode([new_token]),end = "", flush=True)
    print(token_ids)

