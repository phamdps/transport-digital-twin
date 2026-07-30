import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from forecasting.layers.cross_encoder import Encoder
from forecasting.layers.cross_decoder import Decoder
from forecasting.layers.attn import FullAttention, AttentionLayer, TwoStageAttentionLayer
from forecasting.layers.cross_embed import DSW_embedding
from forecasting.layers.RevIN import RevIN
from math import ceil

# Zhang, Y.; and Yan, J. 2023. Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting. In The Eleventh International Conference on Learning Representations

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.data_dim = configs.enc_in
        self.in_len = configs.seq_len
        self.out_len = configs.pred_len
        self.seg_len = configs.seg_len
        self.merge_win = configs.win_size
        self.d_model = configs.d_model

        self.baseline = configs.baseline
        self.factor = configs.cross_factor
        self.win_size = configs.win_size
        self.d_ff = configs.d_ff
        self.n_heads = configs.n_heads
        self.e_layers = configs.e_layers
        self.dropout = configs.dropout
        self.flash_attn = configs.flash_attn # added flash attention as a param
        
        # revin additions
        self.revin = configs.revin
        self.affine= configs.affine
        self.subtract_last = configs.subtract_last

        self.channel_independence = configs.channel_independence
        if self.revin ==1: self.revin_layer = RevIN(configs.enc_in, affine=self.affine, subtract_last=self.subtract_last)


        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * self.in_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(1.0 * self.out_len / self.seg_len) * self.seg_len
        self.in_len_add = self.pad_in_len - self.in_len

        # Embedding
        self.enc_value_embedding = DSW_embedding(self.seg_len, self.d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, self.data_dim, (self.pad_in_len // self.seg_len), self.d_model))
        self.pre_norm = nn.LayerNorm(self.d_model)

        # Encoder
        self.encoder = Encoder(self.e_layers, self.win_size, self.d_model, self.n_heads, self.d_ff, self.flash_attn, block_depth = 1, \
                                    dropout = self.dropout,in_seg_num = (self.pad_in_len // self.seg_len), factor = self.factor \
                                        ,channel_independence=self.channel_independence)
        
        # Decoder
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, self.data_dim, (self.pad_out_len // self.seg_len), self.d_model))
        self.decoder = Decoder(self.seg_len, self.e_layers + 1, self.d_model, self.n_heads, self.d_ff, self.flash_attn, self.dropout, \
                                    out_seg_num = (self.pad_out_len // self.seg_len), factor = self.factor)
        
    def forward(self, x_seq): # 32, 192, 7
        if (self.baseline == 1):
            base = x_seq.mean(dim = 1, keepdim = True)
        else:
            base = 0

        # pdb.set_trace()
        # add revin
        # if self.revin ==1: x_seq = self.revin_layer(x_seq, 'norm')

        batch_size = x_seq.shape[0] 
        if (self.in_len_add != 0):
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.in_len_add, -1), x_seq), dim = 1)

        x_seq = self.enc_value_embedding(x_seq)
        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)
        
        enc_out = self.encoder(x_seq)

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat = batch_size)
        predict_y = self.decoder(dec_in, enc_out)

        out = base + predict_y[:, :self.out_len, :]

        # denorm
        # if self.revin ==1: out = self.revin_layer(out, 'denorm')


        return out