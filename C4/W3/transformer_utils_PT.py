# import os

import numpy as np
#import pandas as pd
import torch
import torch.nn as nn
#import matplotlib.pyplot as plt
#import time
#import utils


device = torch.device("cuda")
print(device)

def positional_encoding(num_positions, d_model): 
    
    position = np.arange(num_positions)[:, np.newaxis]
    k = np.arange(d_model)[np.newaxis, :]
    i = k // 2
    
    angle_rates = 1 / np.power(10000, (2 * i) / np.float32(d_model))
    angle_rads = position * angle_rates
    
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1:2])    
    
    pos_encoding = angle_rads[np.newaxis, ...]
    return torch.tensor(pos_encoding, dtype=torch.float32, device=device)


def create_padding_mask(token_ids):
    seq = torch.logical_not(torch.eq(token_ids, 0)).float()
    return seq


def create_look_ahead_mask(sequence_length, batch_size):
    mask = torch.tril(torch.ones((1 * batch_size, sequence_length, sequence_length)))
    return mask


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = torch.matmul(q, k.T)

    dk = torch.tensor(k.size(-1), dtype=torch.float)
    scaled_attention_logits = matmul_qk / torch.sqrt(dk)
    scaled_attention_logits = torch.unsqueeze(scaled_attention_logits, dim=0)
    
    if mask is not None:
        scaled_attention_logits += (1. - mask) * -1e9
        
    attention_weights = torch.softmax(scaled_attention_logits, dim=-1)
    
    output = torch.matmul(attention_weights, v)
    return output, attention_weights


def FullyConnected(embedding_dim, fully_connected_dim):
    return nn.Sequential(
        nn.Linear(embedding_dim, fully_connected_dim, device=device),
        nn.ReLU(),
        nn.Linear(fully_connected_dim, embedding_dim, device=device)
    )


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, fully_connected_dim, dropout_rate=0.1, layernorm_eps=1e-6):
        super().__init__()
        
        self.mha = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout_rate, batch_first=True, device=device)
        self.ffn = FullyConnected(embedding_dim, fully_connected_dim)
        
        self.layernorm1 = nn.LayerNorm(embedding_dim, eps=layernorm_eps, device=device)
        self.layernorm2 = nn.LayerNorm(embedding_dim, eps=layernorm_eps, device=device)
        
        self.dropout_ffn = nn.Dropout(dropout_rate)
    
    def forward(self, x, mask):
        mha_output, _ = self.mha(x, x, x, mask)
        skip_x_attention = self.layernorm1(x + mha_output)
        
        ffn_output = self.ffn(skip_x_attention)
        ffn_output = self.dropout_ffn(ffn_output)
        encoder_layer_out = self.layernorm2(ffn_output + skip_x_attention)
        
        return encoder_layer_out
    

class Encoder(nn.Module):
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, input_vocab_size, 
                maximum_position_encoding, dropout_rate=0.1, layernorm_eps=1e-6):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_vocab_size, self.embedding_dim)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.embedding_dim).to(device)
        
        self.enc_layers = [EncoderLayer(embedding_dim=self.embedding_dim, 
                                        num_heads=num_heads,
                                        fully_connected_dim=fully_connected_dim,
                                        dropout_rate=dropout_rate,
                                        layernorm_eps=layernorm_eps) 
                           for _ in range(num_layers)]
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x, mask):
        seq_len = x.shape[1]
        x = self.embedding(x)
        x *= torch.sqrt(torch.tensor(self.embedding_dim)).float()
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)

        return x
    

class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, fully_connected_dim, dropout_rate=0.1, layernorm=1e-6):
        super().__init__()
        
        self.mha1 = nn.MultiheadAttention(embedding_dim, num_heads, dropout_rate, batch_first=True, device=device)
        self.mha2 = nn.MultiheadAttention(embedding_dim, num_heads, dropout_rate, batch_first=True, device=device)
        
        self.ffn = FullyConnected(embedding_dim, fully_connected_dim)
        
        self.layernorm1 = nn.LayerNorm(embedding_dim, layernorm, device=device)
        self.layernorm2 = nn.LayerNorm(embedding_dim, layernorm, device=device)
        self.layernorm3 = nn.LayerNorm(embedding_dim, layernorm, device=device)
        
        self.dropout_ffn = nn.Dropout(dropout_rate)
        
    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        
        mult_attn_out1, attn_weights_block1 = self.mha1(x, x, x, attn_mask=look_ahead_mask, average_attn_weights=False)
        Q1 = self.layernorm1(x + mult_attn_out1)
        
        mult_attn_out2, attn_weights_block2 = self.mha2(Q1, enc_output, enc_output, key_padding_mask=padding_mask, average_attn_weights=False)
        mult_attn_out2 = self.layernorm2(mult_attn_out2 + Q1)

        ffn_output = self.ffn(mult_attn_out2)
        ffn_output = self.dropout_ffn(ffn_output)
        out3 = self.layernorm3(ffn_output + mult_attn_out2)

        return out3, attn_weights_block1, attn_weights_block2
    

class Decoder(nn.Module):
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, target_vocab_size,
                maximum_position_encoding, dropout_rate=0.1, layernorm_eps=1e-6):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(target_vocab_size, embedding_dim)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.embedding_dim)

        self.dec_layers = [DecoderLayer(self.embedding_dim,
                                       num_heads,
                                       fully_connected_dim,
                                       dropout_rate,
                                       layernorm_eps)
                           for _ in range(self.num_layers)]
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        seq_len = x.size(1)
        attention_weights = {}
        
        x = self.embedding(x)
        x *= torch.sqrt(torch.tensor(self.embedding_dim, dtype=torch.float))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        
        for i in range(self.num_layers):
            x, attention_weights_block1, attention_weights_block2 = self.dec_layers[i](x,
                                                                                       enc_output,
                                                                                       look_ahead_mask,
                                                                                       padding_mask)
            
            attention_weights['decoder_layer{}_block1_self_att'.format(i+1)] = attention_weights_block1
            attention_weights['decoder_layer{}_block2_decenc_att'.format(i+1)] = attention_weights_block2
        
        return x, attention_weights
    

class Transformer(nn.Module):
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim,
                input_vocab_size, target_vocab_size, max_positional_encoding_input, 
                max_positional_encoding_target, dropout_rate=0.1, layernorm_eps=1e-6):
        super().__init__()
        
        self.encoder = Encoder(num_layers, embedding_dim, num_heads,
                               fully_connected_dim, input_vocab_size,
                               max_positional_encoding_input, dropout_rate, layernorm_eps)
        
        self.decoder = Decoder(num_layers, embedding_dim, num_heads,
                               fully_connected_dim, target_vocab_size, 
                               max_positional_encoding_target, dropout_rate, layernorm_eps)
        
        self.final_layer = nn.Linear(embedding_dim, target_vocab_size)
        
    def forward(self, input_sentence, output_sentence, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(input_sentence, enc_padding_mask)
        
        dec_output, attention_weights = self.decoder(output_sentence, enc_output, look_ahead_mask, dec_padding_mask)
        
        final_out = self.final_layer(dec_output)
        
        if training == False:
            final_out = torch.softmax(final_out, dim=-1)
            
        final_out = torch.transpose(final_out, 1, 2)
        
        return final_out, attention_weights
    

class CustomSchedule():
    def __init__(self, d_model, warmup_steps=4000):
        self.d_model = torch.tensor(d_model, dtype=torch.float32)
        self.warmup_steps = warmup_steps
        
    def get_lr(self, step):
        arg1 = torch.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        
        return torch.rsqrt(self.d_model) * torch.minimum(arg1, arg2)
    

loss_object = nn.CrossEntropyLoss(reduction='none')


def masked_loss(preds, ground_truth):
    mask = torch.logical_not(torch.eq(ground_truth, 0))
    loss_ = loss_object(preds, ground_truth)
    
    loss_ *= mask
    
    return torch.sum(loss_) / torch.sum(mask)


def next_word(model, encoder_input, output):    
    
    enc_padding_mask = create_padding_mask(encoder_input).to(device)
    dec_padding_mask = create_padding_mask(output).to(device=device, dtype=torch.bool)

    model.eval()
    
    predictions = model(
        input_ids=encoder_input, attention_mask=enc_padding_mask, decoder_input_ids=output,
        decoder_attention_mask=dec_padding_mask, return_dict=False)
    
    predictions = torch.transpose(predictions[0], 1, 2)
    predictions = predictions[:, :, -1:] 
    predicted_id = torch.argmax(predictions, dim=1)
    
    return predicted_id