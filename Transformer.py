import torch
import torch.nn as nn
from Attention import MaskedMultiHeadAttention

class MLP(nn.Module):
    def __init__(self, embedding_size = 256, mlp_ratio = 4):
        super().__init__()
        self.embedding_size = embedding_size
        self.mlp_ratio = mlp_ratio
        self.fc1 = nn.Linear(embedding_size, embedding_size*mlp_ratio)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(embedding_size*mlp_ratio, embedding_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class EncoderTransformerLayer(nn.Module): #EncoderLayer, post norm
    def __init__(self, embedding_size = 256, num_heads = 8, mlp_ratio = 4, bias = False):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.AttentionModule = MaskedMultiHeadAttention(input_emb = embedding_size, kq_emb = embedding_size, v_emb = embedding_size, num_heads = num_heads, bias = bias)
        self.LayerNorm1 = nn.LayerNorm(embedding_size)
        self.mlp_ratio = mlp_ratio
        self.mlp = MLP(embedding_size, mlp_ratio)
        self.LayerNorm2 = nn.LayerNorm(embedding_size)

    def forward(self, x):
        x_attn, _ = self.AttentionModule(x, x, x)
        x = x_attn + x
        x = self.LayerNorm1(x)
        hidden_x = self.mlp(x)
        x = hidden_x + x
        x = self.LayerNorm2(x)
        return x

class EncoderTransformer(nn.Module):
    def __init__(self, num_layers = 4, embedding_size = 256, num_heads = 8, mlp_ratio = 4, bias = False):
        super().__init__()
        self.encoderLayers = nn.Sequential(*[EncoderTransformerLayer(embedding_size = embedding_size, num_heads = num_heads, mlp_ratio = mlp_ratio, bias = bias)]*num_layers)
    def forward(self, x):
        for module in self.encoderLayers:
            x = module(x)
        return x



class DecoderTransformerLayer(nn.Module):#OG GPT2 post norm decoder layer- Note that masked multi head and then multi head and then FFN are being used
    def __init__(self, embedding_size = 256, num_heads = 8, is_causal = True, mlp_ratio = 4, bias = False):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.MaskedAttentionModule = MaskedMultiHeadAttention(input_emb = embedding_size, kq_emb = embedding_size, v_emb = embedding_size, num_heads = num_heads, bias = bias)
        self.LayerNorm1 = nn.LayerNorm(embedding_size)
        self.AttentionModule = MaskedMultiHeadAttention(input_emb = embedding_size, kq_emb = embedding_size, v_emb = embedding_size, num_heads = num_heads, bias = bias)
        self.LayerNorm2 = nn.LayerNorm(embedding_size)
        self.mlp_ratio = mlp_ratio
        self.mlp = MLP(embedding_size, mlp_ratio)
        self.LayerNorm3 = nn.LayerNorm(embedding_size)

    def forward(self, q, k, v): # q is generated from the output sequence, k and v are generated from the source sequence
        masked_x_attn = self.MaskedAttentionModule(v, v, v, masked = True)
        x = masked_x_attn + x
        x = self.LayerNorm1(x)
        x_attn = self.AttentionModule(x, x, x)
        x = x_attn + x
        x = self.LayerNorm2(x)
        hidden_x = self.mlp(x)
        x = hidden_x + x
        x = self.LayerNorm3(x)
        return x

class CausalDecoderTransformerLayer(nn.Module): #This is a pre norm implementation
    def __init__(self, embedding_size = 256, num_heads = 8, is_causal = True, mlp_ratio = 4, bias = False):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.LayerNorm1 = nn.LayerNorm(embedding_size)
        self.AttentionModule = MaskedMultiHeadAttention(input_emb = embedding_size, kq_emb = embedding_size, v_emb = embedding_size, num_heads = num_heads, bias = bias)
        self.LayerNorm2 = nn.LayerNorm(embedding_size)
        self.mlp_ratio = mlp_ratio
        self.mlp = MLP(embedding_size, mlp_ratio)

    def forward(self, x):
        x_norm = self.LayerNorm1(x)
        x_attn, _ = self.AttentionModule(x_norm, x_norm, x_norm, masked = True)
        x = x_attn + x
        x_norm = self.LayerNorm2(x)
        hidden_x = self.mlp(x_norm)
        x = hidden_x + x
        return x

class CausalDecoderOnlyTransformer(nn.Module):
    def __init__(self, num_layers = 4, embedding_size = 256, num_heads = 8, mlp_ratio = 4, bias = False):
        super().__init__()
        self.decoderLayers = nn.Sequential(*[CausalDecoderTransformerLayer(embedding_size = embedding_size, num_heads = num_heads, mlp_ratio = mlp_ratio, bias = bias)]*num_layers)
    def forward(self, x):
        for module in self.decoderLayers:
            x = module(x)
        return x


if __name__ == "__main__":
    x = torch.randn(4, 7, 256)
    #A = EncoderTransformer()
    #f = A(x)
    A = CausalDecoderOnlyTransformer()
    f = A(x)
