import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class MHA(nn.Module):

    def __init__(self, d_model, num_heads, dropout = 0.1):

        super().__init__()

        assert d_model % num_heads == 0, 'd_model must be divisible by the number of heads'

        self.dk = d_model // num_heads

        self.scale = self.dk ** -0.5

        self.num_heads = num_heads

        self.W_q = nn.Linear(d_model, d_model, bias= False)
        self.W_k = nn.Linear(d_model, d_model, bias= False)
        self.W_v = nn.Linear(d_model, d_model, bias= False)
        self.W_o = nn.Linear(d_model, d_model)

        self.drop = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask = None):

        N, seq_length, d_model = Q.shape
        N_src, seg_tar, _ = K.shape
        Q = self.W_q(Q).view(N, seq_length, self.num_heads, self.dk).transpose(1,2)
        K = self.W_k(K).view(N_src, seg_tar, self.num_heads, self.dk).transpose(1,2)
        V = self.W_v(V).view(N_src, seg_tar, self.num_heads, self.dk).transpose(1,2)

        attn_scores = (Q @ K.transpose(-2, -1)) * self.scale

        if mask is not None:

            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = torch.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_probs, V)

        output = output.transpose(1,2).contiguous().view(N, seq_length, d_model)

        output = self.drop(self.W_o(output))

        return output


class LinMHA(nn.Module):
    def __init__(self, d_model, num_heads, dropout = 0.1):
        super().__init__()

        assert d_model % num_heads == 0, 'd_model must be divisible by the number of heads'

        self.dk = d_model // num_heads

        self.scale = self.dk ** -0.5

        self.num_heads = num_heads

        self.W_q = nn.Linear(d_model, d_model, bias= False)
        self.W_k = nn.Linear(d_model, d_model, bias= False)
        self.W_v = nn.Linear(d_model, d_model, bias= False)
        self.W_o = nn.Linear(d_model, d_model)

        self.drop = nn.Dropout(dropout)
        
    def forward(self, q, k, v):

        B, seq_length, C = q.shape
        B_src, seg_tar, _ = k.shape
        q = self.W_q(q).view(B, seq_length, self.num_heads, self.dk).transpose(1,2)
        k = self.W_k(k).view(B, seg_tar, self.num_heads, self.dk).transpose(1,2)
        v = self.W_v(v).view(B, seg_tar, self.num_heads, self.dk).transpose(1,2)
        
        G = k.transpose(-2, -1) * self.scale @ v
        
        out = q *  self.scale @ G
        out = self.W_o(out.transpose(1,2).contiguous().view(B, seq_length, C))

        return out



class EncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, mul = 4 ,dropout = 0.1):

        super().__init__()

        self.sa = MHA(d_model, num_heads, dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * mul),
            nn.GELU(),
            nn.Linear(mul * d_model, d_model)
        )

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)



    def forward(self, x):

        x = x + self.ln1(self.sa(x,x, x))
        x = x + self.ln2(self.ffn(x))

        return x



class ViTEncoder(nn.Module):

    def __init__(self, d_model, num_heads, n_blocks, mul = 4,patch_size = 16, img_size = 224, is_conv = False, device = 'cpu'):

        
        super().__init__()

        self.is_conv = is_conv
        
        if self.is_conv:

            self.lp_src = nn.Conv2d(1, d_model, kernel_size=patch_size, stride=patch_size)
        
        else:

            self.lp_src = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
                nn.LayerNorm(patch_size**2),
                nn.Linear(patch_size**2, d_model),
                nn.LayerNorm(d_model),
            ) 


            self.pos = nn.Parameter(torch.randn(1, (img_size // patch_size)**2, d_model, device = device))


        self.blocks = nn.Sequential(
            *[EncoderLayer(d_model, num_heads, mul) for _ in range(n_blocks)]
        )


    def forward(self, x):

        x = self.lp_src(x)

        if self.is_conv:
            x = x.flatten(2)
            x = x.transpose(1,2)

        if not self.is_conv:

            x += self.pos

        for block in self.blocks:

            x = block(x)

        return x


class CnnEncoder(nn.Module):
    
    def __init__(self, model, in_ch = 1, emd_size = 512, is_set_grad = True):

        super().__init__()

        self.model = model(pretrained = True)
        self.model.conv1 = nn.Conv2d(in_ch, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model  = nn.Sequential(*list(self.model.children())[:-2])
        
        
        self.last_conv = nn.Conv2d(2048, emd_size, kernel_size = 1)
        if is_set_grad:
            self.set_grad()
        
    def forward(self, x):
        
        x = self.model(x)
        
        x = self.last_conv(x)
        
        x = x.flatten(2).permute(0, 2, 1)
        
        return x
    
    def set_grad(self):
        
        for param in self.model.parameters():
    
            param.requires_grad_(False)
        
        for param in self.model[0].parameters():
    
            param.requires_grad_(True)
        
        for param in self.last_conv.parameters():
            
            param.requires_grad_(True)

    

        

        




