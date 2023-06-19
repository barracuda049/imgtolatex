import torch.nn as nn

from decoder import Decoder
from encoders import ViTEncoder, CnnEncoder

class Transformer(nn.Module):

    def __init__(self, d_model, num_heads, n_blocks, vocab_size, max_length, 
                 hidden_size = 2048, dropout = 0.1, mul = 4,patch_size = 16, 
                 img_size = 224, is_conv = False, cnn_enc = None, is_set_grad = True, device = 'cpu'):
        
        

        if cnn_enc is not None:
            super().__init__()
            self.encoder = CnnEncoder(cnn_enc, 1, d_model, is_set_grad=is_set_grad)

        else:
            super().__init__()
            self.encoder = ViTEncoder(d_model, num_heads, n_blocks, 
                                      mul, patch_size, img_size, is_conv)
        

        self.decoder = Decoder(vocab_size, max_length,d_model, num_heads,n_blocks,
                               mul, dropout)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size)
        )


    def forward(self, src, tgt):

        src = self.encoder(src)

        tgt = self.decoder(tgt, src)

        output = self.mlp(tgt)

        return output
    
