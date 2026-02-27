import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from xformers.ops import memory_efficient_attention
from xformers.ops.fmha.flash import FwOp,BwOp
import numpy as np
import pdb

from math import sqrt

class FullAttention(nn.Module):
    '''
    The Attention operation
    '''
    def __init__(self, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout) # testing flash attn
        
    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        return V.contiguous()


class FlashAttentionLayer(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    '''
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, dropout = 0.1):
        super(FlashAttentionLayer, self).__init__()

        self.dropout = dropout
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        # self.inner_attention = FullAttention(scale=None, attention_dropout = dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        # requires n_heads 64, 128, 256
        # requires query.shape[-1] % 8 != 0
        # requires float16
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        # pdb.set_trace()

        queries = self.query_projection(queries).view(B, L, H, -1)
        # padding_needed = (8 - (queries.shape[-1] % 8)) % 8
        # padded_queries = F.pad(queries, (0, padding_needed), "constant", 0)
        # [96, 84, 8, 64]
        keys = self.key_projection(keys).view(B, S, H, -1)
        # padded_keys = F.pad(keys, (0,  padding_needed), "constant", 0)

        # [96, 84, 8, 64]
        values = self.value_projection(values).view(B, S, H, -1)
        # padded_values = F.pad(values, (0, padding_needed), "constant", 0)

        # [96, 84, 8, 64]

        # out = self.inner_attention(
        #     queries,
        #     keys,
        #     values,
        # ) # torch.Size([96, 84, 8, 64])
        # out[0,0,0,:]
        # [ 0.1469,  0.0298, -0.0687,  0.2731, -0.0742, -0.0473, -0.2684, -0.0313,
        # -0.1688, -0.0727, -0.0582, -0.1530,  0.0690, -0.0580,  0.0152, -0.0359,
        # -0.1700, -0.0182,  0.1414, -0.0897, -0.0291, -0.2854,  0.0498, -0.2425,
        # -0.1522, -0.1079, -0.0646,  0.3222, -0.3214, -0.1861,  0.2049, -0.2024,
        #  0.1469,  0.0481, -0.4203, -0.1375, -0.0682, -0.1679,  0.2729,  0.2072,
        # -0.0476,  0.0808,  0.0691, -0.0006,  0.4074,  0.0617,  0.0575, -0.1210,
        #  0.2381,  0.0472,  0.2280, -0.0764, -0.1068, -0.1099,  0.2288,  0.1993,
        # -0.1158,  0.1702, -0.0360, -0.2043,  0.1669,  0.3085, -0.3605, -0.0781],

        out = memory_efficient_attention(queries, keys, values, p = self.dropout) # flash attn  op=(FwOp,BwOp)
        # by default it scales
        # [96, 84, 8, 64]
        # out1[0,0,0,:]
        # [ 0.1469,  0.0298, -0.0687,  0.2731, -0.0742, -0.0473, -0.2684, -0.0313,
        # -0.1688, -0.0727, -0.0582, -0.1530,  0.0690, -0.0580,  0.0152, -0.0359,
        # -0.1700, -0.0182,  0.1414, -0.0897, -0.0291, -0.2854,  0.0498, -0.2425,
        # -0.1522, -0.1079, -0.0646,  0.3222, -0.3214, -0.1861,  0.2049, -0.2024,
        #  0.1469,  0.0481, -0.4203, -0.1375, -0.0682, -0.1679,  0.2729,  0.2072,
        # -0.0476,  0.0808,  0.0691, -0.0006,  0.4074,  0.0617,  0.0575, -0.1210,
        #  0.2381,  0.0472,  0.2280, -0.0764, -0.1068, -0.1099,  0.2288,  0.1993,
        # -0.1158,  0.1702, -0.0360, -0.2043,  0.1669,  0.3085, -0.3605, -0.0781],

        # with 0 dropout it matches


        out = out.view(B, L, -1)

        return self.out_projection(out)


class AttentionLayer(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    '''
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, dropout = 0.1):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = FullAttention(scale=None, attention_dropout = dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
        )

        out = out.view(B, L, -1)

        return self.out_projection(out)

class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''
    def __init__(self, seg_num, factor, d_model, n_heads, d_ff = None, flash_attn=0, dropout=0.1, channel_independence=0):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.flash_attn=flash_attn
        print("self.flash_attn : ", self.flash_attn)
        if self.flash_attn == 1:
            self.time_attention = FlashAttentionLayer(d_model, n_heads, dropout = dropout)
            self.dim_sender = FlashAttentionLayer(d_model, n_heads, dropout = dropout)
            self.dim_receiver = FlashAttentionLayer(d_model, n_heads, dropout = dropout)
        else:
            self.time_attention = AttentionLayer(d_model, n_heads, dropout = dropout)
            self.dim_sender = AttentionLayer(d_model, n_heads, dropout = dropout)
            self.dim_receiver = AttentionLayer(d_model, n_heads, dropout = dropout)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))
        
        self.channel_independence = channel_independence

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.GELU(),
                                nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.GELU(),
                                nn.Linear(d_ff, d_model))

    def forward(self, x): # [32, 3, 56, 216]
        #Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model') # bs x C x PN x d_model -> (bs*C) x PN x d_model
        time_enc = self.time_attention(
            time_in, time_in, time_in
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        #Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        if (self.channel_independence == 0) :
            dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b = batch)
            batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat = batch)
            dim_buffer = self.dim_sender(batch_router, dim_send, dim_send)
            dim_receive = self.dim_receiver(dim_send, dim_buffer, dim_buffer)
            dim_enc = dim_send + self.dropout(dim_receive)
            dim_enc = self.norm3(dim_enc)
            dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
            dim_enc = self.norm4(dim_enc)

            final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b = batch)
        else :
            final_out = rearrange(dim_in, '(b ts_d) seg_num d_model -> b ts_d seg_num d_model', b = batch)

        return final_out
