import os
import string
from collections import Counter
from tqdm import tqdm

class BytePairEncoding():
    def __init__(self, vocab_size = 1024):
        self.vocab_size = vocab_size
        self.tokenizer = dict()
        self.ALL_TEXTS = ""
        self.ALL_WORDS = []
        self.WORDS_FREQ_DICT = None
        self.punctuation_list = list(string.punctuation)
        self.punctuation_counter = {k:0 for k in self.punctuation_list}
        self.vocab = set()
        self.reverse_vocab = dict()
        self.word_splits = []
    def get_texts(self, path):
        txts = os.listdir(path)
        for filenames in txts:
            with open(os.path.join(path, filenames)) as file:
                self.ALL_TEXTS += " " + file.read()

    def initialize_vocab(self):
        for keys, values in self.WORDS_FREQ_DICT.items():
            self.word_splits += [[list(keys), values]]
            self.vocab = self.vocab.union(set(list(keys)))
        self.vocab = list(self.vocab)

    def add_vocab(self):
        while(len(self.vocab) < self.vocab_size):
            WORD_SPLIT = dict()
            for i in range(len(self.word_splits)):
                X = self.word_splits[i][0]
                x_len = len(X)
                if(x_len==1):
                    continue
                
                for j in range(x_len-1):
                    this_string = X[j]+X[j+1]
                    val = WORD_SPLIT.get(this_string, 0)   
                    WORD_SPLIT[this_string] = val + self.word_splits[i][1]
            sorted_word_splits = sorted(WORD_SPLIT.items(), key = lambda item: item[1], reverse=True)
            token_added = sorted_word_splits[0][0]
            self.vocab.append(sorted_word_splits[0][0])
            for i in range(len(self.word_splits)):
                X = self.word_splits[i][0]
                x_len = len(X)
                if(x_len==1):
                    continue
                new_X = []
                j = 0 
                while j<(x_len-1):
                    this_string = X[j]+X[j+1]
                    if(this_string == token_added):
                        new_X.append(this_string)
                        j+=1
                    else:
                        new_X.append(X[j])
                    j+=1
                this_string = X[x_len-2] + X[x_len-1]
                if(this_string != token_added):
                    new_X.append(X[x_len-1])
                self.word_splits[i][0] = new_X
            print(self.vocab)
        final_vocab = dict()
        for i, j in enumerate(self.vocab):
            final_vocab[j] = i
            self.reverse_vocab[i] = j
        self.vocab = final_vocab
        

    def normalize_texts(self, text):
        for j in self.punctuation_list:
            self.punctuation_counter[j] += text.count(j)
            text = text.replace(j, " ")
        text = text.replace("\n", " ")
        text = text.replace("\r", " ")
        text = text.lower()
        text = text.split()
        text_split = [i+'_' for i in text]
        return text, text_split

    def generate_freq_dict(self):
        self.WORDS_FREQ_DICT = dict(Counter(self.ALL_WORDS))
    
    def per_word_tokens(self, text):
        if(text in self.vocab):
            return [self.vocab[text]]
        i = len(text)-1
        while(i>0):
            if(text[:i] in self.vocab):
                return [self.vocab[text[:i]]] + self.per_word_tokens(text[i:])
            i-=1

    def encoding(self, text):
        text, text_split = self.normalize_texts(text)
        #breakpoint()
        tokens = []
        for t in text_split:
            text_tokens = self.per_word_tokens(t)
            tokens.extend(text_tokens)
        return tokens
            
    def decoding(self, tokens):
        return [self.reverse_vocab[i] for i in tokens]

    def run_byte_pair(self, path):
        self.get_texts(path)
        self.ALL_TEXTS, self.ALL_WORDS = self.normalize_texts(self.ALL_TEXTS)
        self.generate_freq_dict()
        self.initialize_vocab()
        self.add_vocab()

def save_tokenizer(self, path):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(path, f)

if __name__ == "__main__":
    tokenizer = BytePairEncoding(vocab_size = 1024)
    tokenizer.run_byte_pair("/home/cv-research/Transformer/shakespeare-dataset/text")
    tokens = tokenizer.encoding("Hello my name is Anthony Gonzalves")
    breakpoint()
