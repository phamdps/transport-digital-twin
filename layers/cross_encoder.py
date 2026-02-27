import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from layers.attn import FullAttention, AttentionLayer, TwoStageAttentionLayer
# , FullAttention_full, AttentionLayer_full, TwoStageAttentionLayer_full (kiran: not sure why? missing in attn)
from math import ceil

class SegMerging(nn.Module):
    '''
    Segment Merging Layer.
    The adjacent `win_size' segments in each dimension will be merged into one segment to
    get representation of a coarser scale
    we set win_size = 2 in our paper
    '''
    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size * d_model, d_model)
        self.norm = norm_layer(win_size * d_model)

    def forward(self, x):
        """
        x: B, ts_d, L, d_model
        """
        batch_size, ts_d, seg_num, d_model = x.shape
        pad_num = seg_num % self.win_size
        if pad_num != 0: 
            pad_num = self.win_size - pad_num
            x = torch.cat((x, x[:, :, -pad_num:, :]), dim = -2)

        seg_to_merge = []
        for i in range(self.win_size):
            seg_to_merge.append(x[:, :, i::self.win_size, :])
        x = torch.cat(seg_to_merge, -1)  # [B, ts_d, seg_num/win_size, win_size*d_model]

        x = self.norm(x)
        x = self.linear_trans(x)

        return x

class scale_block(nn.Module):
    '''
    We can use one segment merging layer followed by multiple TSA layers in each scale
    the parameter `depth' determines the number of TSA layers used in each scale
    We set depth = 1 in the paper
    '''
    def __init__(self, win_size, d_model, n_heads, d_ff, flash_attn, depth, dropout, \
                    seg_num = 10, factor=10, channel_independence=0):
        super(scale_block, self).__init__()

        if (win_size > 1):
            self.merge_layer = SegMerging(d_model, win_size, nn.LayerNorm)
        else:
            self.merge_layer = None
        
        self.encode_layers = nn.ModuleList()

        for i in range(depth):
            self.encode_layers.append(TwoStageAttentionLayer(seg_num, factor, d_model, n_heads, \
                                                        d_ff, flash_attn, dropout, channel_independence))
    
    def forward(self, x):
        _, ts_dim, _, _ = x.shape

        if self.merge_layer is not None:
            x = self.merge_layer(x)
        
        for layer in self.encode_layers:
            x = layer(x)        
        
        return x

class Encoder(nn.Module):
    '''
    The Encoder of Crossformer.
    '''
    def __init__(self, e_blocks, win_size, d_model, n_heads, d_ff, flash_attn, block_depth, dropout,
                in_seg_num = 10, factor=10, channel_independence=0):
        super(Encoder, self).__init__()
        self.encode_blocks = nn.ModuleList()

        self.encode_blocks.append(scale_block(1, d_model, n_heads, d_ff, flash_attn, block_depth, dropout,\
                                            in_seg_num, factor, channel_independence))
        for i in range(1, e_blocks):
            self.encode_blocks.append(scale_block(win_size, d_model, n_heads, d_ff, flash_attn, block_depth, dropout,\
                                            ceil(in_seg_num/win_size**i), factor, channel_independence))

    def forward(self, x):
        encode_x = []
        encode_x.append(x)
        
        for block in self.encode_blocks:
            x = block(x)
            encode_x.append(x)

        return encode_x
    

class Encoder_Light(nn.Module):
   
    def __init__(self, e_blocks, win_size, d_model, n_heads, d_ff, flash_attn, block_depth, dropout,
                 in_seg_num=10, factor=10, channel_independence=0, final_seg_num=None):
        super(Encoder_Light, self).__init__()
        self.e_blocks = e_blocks
        self.d_model = d_model

        self.seg_nums = []
        current_seg_num = in_seg_num
        self.seg_nums.append(current_seg_num) 

        for i in range(1, e_blocks+1):
            if i == 1:
                pass
            else:
                current_seg_num = ceil(in_seg_num / (win_size ** (i - 1)))
            self.seg_nums.append(current_seg_num)

        self.encode_blocks = nn.ModuleList()
        self.encode_blocks.append(
            scale_block(1, d_model, n_heads, d_ff, flash_attn, block_depth, dropout,
                        seg_num=self.seg_nums[1], factor=factor, channel_independence=channel_independence)
        )

        for i in range(2, e_blocks+1):
            self.encode_blocks.append(
                scale_block(win_size, d_model, n_heads, d_ff, flash_attn, block_depth, dropout,
                            seg_num=self.seg_nums[i], factor=factor, channel_independence=channel_independence)
            )

        # for residual connections
        self.scale_transformers = nn.ModuleList()
        
        if final_seg_num is None:
            # Dynamically compute `final_seg_num`
            for i in range(1, e_blocks+1):
                if i == 1:
                    pass
                else:
                    current_seg_num = ceil(in_seg_num / (win_size ** (i - 1)))
                self.seg_nums.append(current_seg_num)
            final_seg_num = self.seg_nums[-1]  # Set the computed final segment number
        else:
            for i in range(1, e_blocks):
                current_seg_num = ceil(final_seg_num * (win_size ** (e_blocks - i - 1)))
                self.seg_nums.insert(1, current_seg_num)
            self.seg_nums.append(final_seg_num)
        self.final_seg_num = final_seg_num  # Store for later reference
      
        for i in range(e_blocks):
         
            in_seg_num = self.seg_nums[i]
            out_seg_num = self.seg_nums[-1]
            if in_seg_num == out_seg_num:
           
                self.scale_transformers.append(nn.Identity())
            else:
             
                self.scale_transformers.append(
                    nn.Linear(in_seg_num, out_seg_num)
                )

     
    # Original Forward
    
    def forward(self, x):
        # x: B, ts_d, seg_num, d_model 
        scale_outputs = [x]

        for block in self.encode_blocks:
            x = block(x)
            scale_outputs.append(x)

  
        final_output = scale_outputs[-1]  # B, ts_d, final_seg_num, d_model
        B, ts_d, final_seg_num, d_model = final_output.shape
        for i, out_scale in enumerate(scale_outputs[:-1]):
            seg_num_in = out_scale.shape[2]
            if seg_num_in == final_seg_num:
                final_output = final_output + out_scale
            else:
                #
                # out_scale: B, ts_d, seg_num_in, d_model
                reshape_in = out_scale.reshape(-1, seg_num_in)
      
                transformed = self.scale_transformers[i](reshape_in)  # -1, final_seg_num
                transformed = transformed.reshape(B, ts_d, final_seg_num, d_model)
                final_output = final_output + transformed
                

        return final_output
    
    
    '''
     
     Just stacking and sum version
     
    def forward(self, x):
        # x is shape: [B, ts_d, seg_num, d_model]
        scale_outputs = [x]
        for block in self.encode_blocks:
            x = block(x)
            scale_outputs.append(x)

        # final_output is [B, ts_d, final_seg_num, d_model]
        final_output = scale_outputs[-1]
        B, ts_d, final_seg_num, d_model = final_output.shape

        # Transform each intermediate scale to match final_seg_num in width
        # and accumulate them by stacking and summation.
        transformed_scales = []
        for i, out_scale in enumerate(scale_outputs[:-1]):
            seg_num_in = out_scale.shape[2]
            
            if seg_num_in == final_seg_num:
                # No change needed if segment sizes already match
                transformed_scales.append(out_scale)
            else:
                # Reshape from (B, ts_d, seg_num_in, d_model) -> (B, ts_d, seg_num_in*d_model)
                reshaped = out_scale.view(B, ts_d, seg_num_in * d_model)
                
                # Pass through a learned projection that outputs (B, ts_d, final_seg_num*d_model)
                projected = self.scale_transformers[i](reshaped)
                
                # Reshape back to (B, ts_d, final_seg_num, d_model)
                projected = projected.view(B, ts_d, final_seg_num, d_model)
                transformed_scales.append(projected)

        # Stack all intermediate scales: shape = [num_scales, B, ts_d, final_seg_num, d_model]
        stacked_scales = torch.stack(transformed_scales, dim=0)
        # Sum across the first dimension to get back to [B, ts_d, final_seg_num, d_model]
        sum_scales = stacked_scales.sum(dim=0)
        
        # Add the accumulated result to the final output
        final_output = final_output + sum_scales

        return final_output
    
    
    
    '''
    

        