import torch
import numpy as np
import torch.nn as nn

def positional_encoding(length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :] / depth

    angle_rates = 1 / (10000**depths)
    angle_rads = positions * angle_rates 

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1
    )

    return torch.tensor(pos_encoding, dtype=torch.float32)


class PositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_length=2048):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_length = max_length

        self.embedding = nn.Embedding(self.vocab_size, self.d_model, padding_idx=0)
        self.pos_encoding = positional_encoding(self.max_length, depth=self.d_model)

    def compute_mask(self, x):
        return (x != 0).float()
    
    def forward(self, x):
        mask = self.compute_mask(x)
        length = x.size(1)
        x *= torch.sqrt(self.d_model).float() #should be torch.float32
        x += self.pos_encoding[None, :length, :]
        x *= mask
        return x
    

class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = dropout

    def forward(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        return inputs