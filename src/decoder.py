import torch
import torch.nn as nn
from encoders import MHA

class Decoderlayer(nn.Module):

    
    def __init__(self, d_model, num_heads, mul = 4, dropout = 0.1, device = 'cpu'):
        
        super().__init__()
        
        self.cross_att = MHA(d_model, num_heads)
        self.masked_att = MHA(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, mul * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mul * d_model, d_model),
            
        )

        self.device = device

    def forward(self, x, enc_output):
        
        N, seq_length, _ = x.shape
        
        masked_att = self.masked_att(x, x, x, mask = torch.tril(torch.ones(seq_length, seq_length)).expand(N, 1, seq_length, seq_length).to(self.device))
        x = self.norm1(x + self.dropout(masked_att))
        
        cross_mask = self.cross_att(x, enc_output, enc_output)
        x = self.norm2(x + self.dropout(cross_mask))
        
        ff_out = self.ffn(x)
        
        x = self.norm3(x + self.dropout(ff_out))
        
        return x
    

class Decoder(nn.Module):

    def __init__(self, vocab_size, max_length, d_model, num_heads, n_blocks, mul = 4, dropout = 0.1,device = 'cpu'):

        super().__init__()
        
        self.device = device
        self.lp = nn.Embedding(vocab_size, d_model)

        self.pos = nn.Embedding(max_length, d_model)

        self.blocks = nn.Sequential(
            *[Decoderlayer(d_model, num_heads, mul, dropout, device) for _ in range(n_blocks)] 
        )

    def forward(self, x, enc_output):

        x = self.lp(x)

        x += self.pos(torch.arange(x.shape[1], device = self.device).expand(x.shape[0], x.shape[1]))
        
        for block in self.blocks:
            x = block(x, enc_output)


        return x