import torch
import torch.nn as nn

def apply_absolute_positional_encoding(x, N = 10000):
    batch_size, seq_length, input_emb_dim = x.size()
    assert input_emb_dim > 1
    pos_enc = torch.zeros(seq_length, input_emb_dim, device = x.device)
    dim = torch.arange(0, input_emb_dim//2)
    dim = dim.unsqueeze(0)
    pos = torch.arange(0, seq_length)
    pos = pos.unsqueeze(1)
    cos = torch.cos(pos/(float(N)**((2*dim)/input_emb_dim)))
    sin = torch.sin(pos/(float(N)**((2*dim)/input_emb_dim)))
    pos_enc[:, 0::2] = sin
    pos_enc[:, 1::2] = cos
    x = x + pos_enc
    return x, pos_enc

def apply_rope(x, N = 10000):
    batch_size, seq_length, input_emb_dim = x.size()
    x1 = x.clone()
    pos = torch.arange(0, seq_length, device = x.device)
    pos = pos.unsqueeze(1)
    dim = torch.arange(0, input_emb_dim//2, device = x.device)
    dim = dim.unsqueeze(0)
    cos = torch.cos(pos*(N**(-2*dim/input_emb_dim)))
    sin = torch.sin(pos*(N**(-2*dim/input_emb_dim)))
    x[:, :, 0::2] = x1[:, :, 0::2]*cos - x1[:, :, 1::2]*sin
    x[:, :, 1::2] = x1[:, :, 0::2]*sin + x1[:, :, 1::2]*cos
    return x

class SelfAttention(nn.Module):
    def __init__(self, input_emb, kq_emb, v_emb):
        super().__init__()
        self.kq_emb = kq_emb
        self.query_proj = nn.Linear(input_emb, kq_emb, bias = False)
        self.key_proj = nn.Linear(input_emb, kq_emb, bias = False)
        self.v_proj = nn.Linear(input_emb, v_emb, bias = False)

    def forward(self, x):
        Q = self.query_proj(x) # [num_of_input_tokens, input_emb] -> [num_of_input_tokens, kq_emb]
        K = self.key_proj(x) # [num_of_input_tokens, input_emb] -> [num_of_input_tokens, kq_emb]
        V = self.v_proj(x) # [num_of_input_tokens, input_emb] -> [num_of_input_tokens, v_emb]

        QKt = (Q @ (K.transpose(-1, -2))) * (1 / self.kq_emb**0.5)
        Softie = torch.softmax(QKt, dim = -1)
        output = Softie @ V
        return output

class MaskedSelfAttention(nn.Module):
    def __init__(self, input_emb, kq_emb, v_emb):
        super().__init__()
        self.kq_emb = kq_emb
        self.query_proj = nn.Linear(input_emb, kq_emb, bias = False)
        self.key_proj = nn.Linear(input_emb, kq_emb, bias = False)
        self.v_proj = nn.Linear(input_emb, v_emb, bias = False)

    def forward(self, x):
        Q = self.query_proj(x) # [num_of_input_tokens, input_emb] -> [num_of_input_tokens, kq_emb]
        K = self.key_proj(x) # [num_of_input_tokens, input_emb] -> [num_of_input_tokens, kq_emb]
        V = self.v_proj(x) # [num_of_input_tokens, input_emb] -> [num_of_input_tokens, v_emb]

        QKt = (Q @ (K.transpose(-1, -2))) * (1 / self.kq_emb**0.5)
        mask = torch.ones_like(QKt, dtype = torch.bool).triu(diagonal=1)
        QKt[mask] = float(-1e9)

        breakpoint()
        Softie = torch.softmax(QKt, dim = -1)
        output = Softie @ V
        return output



class MaskedMultiHeadAttention(nn.Module):#I will include batch in this
    def __init__(self, input_emb, kq_emb, v_emb, num_heads, bias = False):
        assert kq_emb%num_heads == 0
        assert v_emb%num_heads == 0
        super().__init__()
        self.kq_emb = kq_emb
        self.v_emb = v_emb
        self.num_heads = num_heads
        self.kq_head_dim = kq_emb // num_heads 
        self.v_head_dim = v_emb // num_heads
        self.query_proj = nn.Linear(input_emb, kq_emb, bias = bias)
        self.key_proj = nn.Linear(input_emb, kq_emb, bias = bias)
        self.value_proj = nn.Linear(input_emb, v_emb, bias = bias)
        self.output_proj = nn.Linear(v_emb, v_emb, bias = bias)


    def forward(self, q, k, v, masked=False):
        batch_size, seq_length, input_emb = q.size()
        Q = self.query_proj(q) #[B, seq_length, input_emb] --> [B, seq_length, kq_emb]
        K = self.key_proj(k) #[B, seq_length, input_emb] --> [B, seq_length, kq_emb]
        V = self.value_proj(v) #[B, seq_length, input_emb] --> [B, seq_length, v_emb]
        Q = Q.view(batch_size, seq_length, self.num_heads, self.kq_head_dim).transpose(2,1) # [B, seq_length, num_heads, kq_head_dim]---->[B, num_heads, seq_length, kq_head_dim]
        K = K.view(batch_size, seq_length, self.num_heads, self.kq_head_dim).transpose(2,1) # [B, seq_length, num_heads, kq_head_dim]---->[B, num_heads, seq_length, kq_head_dim]
        V = V.view(batch_size, seq_length, self.num_heads, self.v_head_dim).transpose(2,1) # [B, seq_length, num_heads, v_head_dim]---->[B, num_heads, seq_length, v_head_dim]
        attention_scores = (Q @ (K.transpose(-1,-2))) * (1 / self.kq_head_dim**0.5) # [B, num_heads, seq_length, seq_length]
        if masked:
            mask = torch.ones_like(attention_scores, dtype = torch.bool).triu(diagonal=1)
            attention_scores = attention_scores.masked_fill(mask, float('-inf'))
        attention_weights = torch.softmax(attention_scores, dim = -1) # [B, num_heads, seq_length, seq_length]

        attention_out = attention_weights @ V # [B, num_heads, seq_length, v_head_length]
        back_to_og = attention_out.transpose(2,1).contiguous().view(batch_size, seq_length, self.v_head_dim*self.num_heads) # [B, seq_length, num_heads * v_head_length]

        output = self.output_proj(back_to_og) # [B, seq_length, num_heads * v_head_length]
        return output, attention_out



if __name__ == "__main__":
    #X = MaskedSelfAttention(4,6,8)
    #a = torch.ones([12, 4])
    #f = X(a)

    a = torch.rand(4, 7, 10) #[batch_size, seq_length, input_emb_dim]
    multihead = MaskedMultiHeadAttention(input_emb = 10, kq_emb = 12, v_emb = 16, num_heads = 2)
    output = multihead(a, a, a, masked=True)
    out, pos = apply_absolute_positional_encoding(a)
    #a1 = apply_rope(a)
    breakpoint()
