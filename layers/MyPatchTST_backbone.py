__all__ = ['PatchTST_backbone', 'CTPatchTST_backbone', 'MultiScaleCTPatchTST_backbone']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

#from collections import OrderedDict
from layers.PatchTST_layers import *
from layers.RevIN import RevIN

# Cell
class PatchTST_backbone(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024, 
                 n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False,
                 verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            patch_num += 1
        
        # Backbone 
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head: 
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten': 
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
        
    
    def forward(self, z):                                                                   # z: [bs x nvars x seq_len]
        # norm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)
            
        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0,1,3,2)                                                              # z: [bs x nvars x patch_len x patch_num]
        
        # model
        z = self.backbone(z)                                                                # z: [bs x nvars x d_model x patch_num]
        z = self.head(z)                                                                    # z: [bs x nvars x target_window] 
        
        # denorm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)
        return z
    
    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


class TargetFusionHead(nn.Module):
    """Target-oriented head for multivariate-input single-target forecasting.

    It combines the target channel representation with a pooled context from
    all channels, then predicts only the target variable.
    """

    def __init__(self, d_model, patch_num, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(2 * d_model * patch_num, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x, target_idx):
        # x: [B, C, D, P]
        target_feat = x[:, target_idx, :, :]        # [B, D, P]
        global_context = x.mean(dim=1)              # [B, D, P]
        fusion = torch.cat([target_feat, global_context], dim=1)
        fusion = self.flatten(fusion)
        fusion = self.linear(fusion)
        fusion = self.dropout(fusion)
        return fusion.unsqueeze(1)                  # [B, 1, target_window]


class TrendResidualFusionHead(nn.Module):
    """Target head that decomposes output into smooth trend and local residual."""

    def __init__(self, d_model, patch_num, target_window, head_dropout=0,
                 trend_smooth_kernel=5, residual_scale=0.1):
        super().__init__()
        if trend_smooth_kernel < 1 or trend_smooth_kernel % 2 == 0:
            raise ValueError("trend_smooth_kernel must be a positive odd integer.")

        self.flatten = nn.Flatten(start_dim=-2)
        self.trend_head = nn.Linear(2 * d_model * patch_num, target_window)
        self.residual_head = nn.Linear(2 * d_model * patch_num, target_window)
        self.dropout = nn.Dropout(head_dropout)
        self.trend_smooth_kernel = trend_smooth_kernel
        self.residual_scale = residual_scale

    def _smooth_trend(self, trend):
        if self.trend_smooth_kernel == 1:
            return trend
        pad = self.trend_smooth_kernel // 2
        trend = F.pad(trend.unsqueeze(1), (pad, pad), mode='replicate')
        trend = F.avg_pool1d(trend, kernel_size=self.trend_smooth_kernel, stride=1)
        return trend.squeeze(1)

    def forward(self, x, target_idx):
        # x: [B, C, D, P]
        target_feat = x[:, target_idx, :, :]
        global_context = x.mean(dim=1)
        fusion = torch.cat([target_feat, global_context], dim=1)
        fusion = self.flatten(fusion)

        trend = self._smooth_trend(self.trend_head(fusion))
        residual = self.residual_head(fusion)
        out = trend + self.residual_scale * residual
        out = self.dropout(out)
        return out.unsqueeze(1)                     # [B, 1, target_window]


class InertiaResidualFusionHead(nn.Module):
    """Target head that predicts a correction over the latest target value."""

    def __init__(self, d_model, patch_num, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.residual_head = nn.Linear(2 * d_model * patch_num, target_window)
        self.dropout = nn.Dropout(head_dropout)
        self.target_window = target_window

    def _build_inertia_baseline(self, target_history):
        # target_history is in the same normalized space as the head output: [B, seq_len]
        last = target_history[:, -1:]
        return last.expand(-1, self.target_window)

    def forward(self, x, target_idx, target_history):
        # x: [B, C, D, P], target_history: [B, seq_len]
        target_feat = x[:, target_idx, :, :]
        global_context = x.mean(dim=1)
        fusion = torch.cat([target_feat, global_context], dim=1)
        fusion = self.flatten(fusion)

        baseline = self._build_inertia_baseline(target_history)
        residual = self.dropout(self.residual_head(fusion))
        return (baseline + residual).unsqueeze(1)    # [B, 1, target_window]


class FaultClassificationHead(nn.Module):
    """Fault type head attached to the shared PatchTST representation."""

    def __init__(self, d_model, num_classes, hidden_dim=None, head_dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or d_model
        self.net = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Dropout(head_dropout),
            nn.Linear(2 * d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, target_idx):
        # x: [B, C, D, P]. Combine target-channel state with global context.
        target_feat = x[:, target_idx, :, :].mean(dim=-1)   # [B, D]
        global_context = x.mean(dim=(1, 3))                 # [B, D]
        return self.net(torch.cat([target_feat, global_context], dim=-1))
        
        
    
    
class TSTiEncoder(nn.Module):  #i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):
        
        
        super().__init__()
        
        self.patch_num = patch_num
        self.patch_len = patch_len
        
        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)

        
    def forward(self, x) -> Tensor:                                              # x: [bs x nvars x patch_len x patch_num]
        
        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0,1,3,2)                                                   # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)                                                          # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))      # u: [bs * nvars x patch_num x d_model]
        u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars x patch_num x d_model]

        # Encoder
        z = self.encoder(u)                                                      # z: [bs * nvars x patch_num x d_model]
        z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))                # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x patch_num]
        
        return z    
            
            
    
# Cell
class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output



class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src




class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


# ==================== Channel-Time PatchTST Backbone ====================

from layers.MyPatchTST_layers import PatchwiseChannelAttention, positional_encoding, Transpose, get_activation_fn


class CTEncoderLayer(nn.Module):
    """Channel-Time encoder layer: Channel Attention -> Time Attention -> FFN.

    Flow: [B, C, P, D]
      1. PatchwiseChannelAttention on [B, C, P, D] (cross-channel fusion)
      2. reshape to [B*C, P, D], standard time self-attention
      3. FFN
    """

    def __init__(self, d_model, n_vars, n_heads_channel=4, n_heads_time=8,
                 d_ff=256, attn_dropout=0., dropout=0., activation="gelu",
                 norm='BatchNorm', res_attention=False, pre_norm=False,
                 use_channel_attention=True):
        super().__init__()
        self.n_vars = n_vars
        self.pre_norm = pre_norm
        self.use_channel_attention = use_channel_attention

        # 1. Channel attention
        if self.use_channel_attention:
            self.channel_attn = PatchwiseChannelAttention(
                d_model, n_heads=n_heads_channel,
                attn_dropout=attn_dropout, proj_dropout=dropout
            )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # 2. Time attention (reuses original _MultiheadAttention)
        self.time_attn = _MultiheadAttention(
            d_model, n_heads_time,
            attn_dropout=attn_dropout, proj_dropout=dropout,
            res_attention=False
        )
        if "batch" in norm.lower():
            self.norm2 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # 3. FFN
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        if "batch" in norm.lower():
            self.norm3 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

        self.res_attention = res_attention

    def forward(self, x, return_attn=False):
        """x: [B, C, P, D]"""
        B, C, P, D = x.shape

        # --- 1. Channel Attention ---
        if self.use_channel_attention:
            if self.pre_norm:
                x_ch = self.channel_attn(self.norm1(x), return_attn=return_attn)
            else:
                x_ch = self.channel_attn(x, return_attn=return_attn)

            if return_attn:
                x_ch, ch_attn = x_ch
            else:
                ch_attn = None

            if self.pre_norm:
                x = x + self.dropout1(x_ch)
            else:
                x = self.norm1(x + self.dropout1(x_ch))
        else:
            ch_attn = None

        # --- 2. Time Attention ---
        # reshape [B, C, P, D] -> [B*C, P, D]
        x_time = x.reshape(B * C, P, D)
        if self.pre_norm:
            x_out, _ = self.time_attn(self.norm2(x_time))
        else:
            x_out, _ = self.time_attn(x_time)

        if self.pre_norm:
            x_time = x_time + self.dropout2(x_out)
        else:
            x_time = self.norm2(x_time + self.dropout2(x_out))

        # --- 3. FFN ---
        if self.pre_norm:
            x_time = x_time + self.dropout3(self.ff(self.norm3(x_time)))
        else:
            x_time = self.norm3(x_time + self.dropout3(self.ff(x_time)))

        # reshape back: [B*C, P, D] -> [B, C, P, D]
        x = x_time.reshape(B, C, P, D)

        if return_attn:
            return x, ch_attn
        return x


class CTEncoder(nn.Module):
    """Channel-Time Encoder: patch embedding -> N x CTEncoderLayer -> output.

    Input:  [B, C, patch_len, patch_num]
    Output: [B, C, d_model, patch_num]
    """

    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=8, n_heads_channel=4,
                 d_ff=256, attn_dropout=0., dropout=0., activation="gelu",
                 norm='BatchNorm', res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, use_channel_attention=True, **kwargs):
        super().__init__()

        self.patch_num = patch_num
        self.n_vars = c_in
        self.d_model = d_model

        # Patch embedding
        self.W_P = nn.Linear(patch_len, d_model)

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, patch_num, d_model)

        # Channel embedding: learnable ID for each variable
        self.channel_embedding = nn.Parameter(torch.randn(1, c_in, 1, d_model) * 0.02)

        self.dropout = nn.Dropout(dropout)

        # Encoder layers
        self.layers = nn.ModuleList([
            CTEncoderLayer(
                d_model=d_model, n_vars=c_in,
                n_heads_channel=n_heads_channel, n_heads_time=n_heads,
                d_ff=d_ff, attn_dropout=attn_dropout, dropout=dropout,
                activation=activation, norm=norm,
                res_attention=res_attention, pre_norm=pre_norm,
                use_channel_attention=use_channel_attention
            ) for _ in range(n_layers)
        ])

    def forward(self, x, return_attn=False):
        """
        x: [B, C, patch_len, patch_num]
        Returns: [B, C, d_model, patch_num]
        """
        B = x.shape[0]

        # Patch embedding: [B, C, patch_len, patch_num] -> [B, C, patch_num, d_model]
        x = x.permute(0, 1, 3, 2)
        x = self.W_P(x)

        # Add positional encoding + channel embedding
        x = self.dropout(x + self.W_pos + self.channel_embedding)

        # Encoder layers
        ch_attn = None
        for i, layer in enumerate(self.layers):
            if return_attn and i == len(self.layers) - 1:
                x, ch_attn = layer(x, return_attn=True)
            else:
                x = layer(x)

        # [B, C, patch_num, d_model] -> [B, C, d_model, patch_num]
        x = x.permute(0, 1, 3, 2)

        if return_attn:
            return x, ch_attn
        return x


class CTPatchTST_backbone(nn.Module):
    """Channel-Time PatchTST backbone.

    Flow: RevIN -> Patching -> CTEncoder -> Flatten_Head -> RevIN^-1

    Input:  [B, C, seq_len]
    Output: [B, C, pred_len]
    """

    def __init__(self, c_in, context_window, target_window, patch_len, stride,
                 max_seq_len=1024, n_layers=3, d_model=128, n_heads=8,
                 n_heads_channel=4, d_ff=256, norm='BatchNorm',
                 attn_dropout=0., dropout=0., activation="gelu",
                 res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True,
                 fc_dropout=0., head_dropout=0, padding_patch=None,
                 individual=False, revin=True, affine=True, subtract_last=False,
                 verbose=False, use_channel_attention=True,
                 use_single_global_channel_attention=False,
                 use_fault_class_head=False, num_fault_classes=5,
                 fault_head_hidden=None, fault_head_dropout=None, **kwargs):
        super().__init__()

        self.use_single_global_channel_attention = use_single_global_channel_attention
        self.use_fault_class_head = use_fault_class_head
        self.target_idx = c_in - 1

        # RevIN
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        # Encoder
        self.encoder = CTEncoder(
            c_in=c_in, patch_num=patch_num, patch_len=patch_len,
            max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
            n_heads=n_heads, n_heads_channel=n_heads_channel, d_ff=d_ff,
            attn_dropout=attn_dropout, dropout=dropout, activation=activation,
            norm=norm, res_attention=res_attention, pre_norm=pre_norm,
            pe=pe, learn_pe=learn_pe, use_channel_attention=use_channel_attention,
            **kwargs
        )

        if self.use_single_global_channel_attention:
            self.global_channel_attn = PatchwiseChannelAttention(
                d_model, n_heads=n_heads_channel,
                attn_dropout=attn_dropout, proj_dropout=dropout
            )
            self.global_channel_norm = nn.LayerNorm(d_model)
            self.global_channel_dropout = nn.Dropout(dropout)

        # Head
        self.head_nf = d_model * patch_num
        self.head = Flatten_Head(individual, c_in, self.head_nf, target_window, head_dropout=head_dropout)
        if self.use_fault_class_head:
            self.fault_head = FaultClassificationHead(
                d_model=d_model,
                num_classes=num_fault_classes,
                hidden_dim=fault_head_hidden,
                head_dropout=head_dropout if fault_head_dropout is None else fault_head_dropout,
            )

    def forward(self, z, return_attn=False):
        """z: [B, C, seq_len]"""
        # RevIN norm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1)

        # Patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z = z.permute(0, 1, 3, 2)  # [B, C, patch_len, patch_num]

        # Encode
        if return_attn:
            z, ch_attn = self.encoder(z, return_attn=True)
        else:
            ch_attn = None
            z = self.encoder(z)

        if self.use_single_global_channel_attention:
            z_for_attn = z.permute(0, 1, 3, 2)
            if return_attn:
                z_attn, global_attn = self.global_channel_attn(z_for_attn, return_attn=True)
                ch_attn = global_attn
            else:
                z_attn = self.global_channel_attn(z_for_attn)
            z = self.global_channel_norm(
                z_for_attn + self.global_channel_dropout(z_attn)
            ).permute(0, 1, 3, 2)

        fault_logits = self.fault_head(z, self.target_idx) if self.use_fault_class_head else None

        # Head
        z = self.head(z)

        # RevIN denorm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0, 2, 1)

        if return_attn:
            if self.use_fault_class_head:
                return z, fault_logits, ch_attn
            return z, ch_attn
        if self.use_fault_class_head:
            return z, fault_logits
        return z


class MultiScaleCTPatchTST_backbone(nn.Module):
    """Multi-scale Channel-Time PatchTST backbone.

    Variables are grouped by physical role for PVCable data, or by a generic
    even split for public datasets without domain-specific channel semantics.
    Each group is encoded with a different patch length/stride, aligned on the
    patch axis, restored to the original channel order, globally fused across
    all channels, then passed to a target-oriented fusion head.
    """

    @staticmethod
    def _build_generic_group_configs(c_in):
        """Build non-empty generic groups for datasets without a PVCable layout."""
        if c_in < 1:
            raise ValueError("c_in must be a positive integer.")

        target_idx = c_in - 1
        context_indices = list(range(target_idx))
        q, r = divmod(len(context_indices), 3)
        sizes = [q + (1 if i < r else 0) for i in range(3)]

        chunks = []
        start = 0
        for size in sizes:
            chunks.append(context_indices[start:start + size])
            start += size

        chunks[2].append(target_idx)
        groups = [
            ("short_response", chunks[0]),
            ("environmental", chunks[1]),
            ("trend", chunks[2]),
        ]
        return [(name, indices) for name, indices in groups if indices]

    def __init__(self, c_in, context_window, target_window,
                 fast_patch_len=8, fast_stride=4,
                 medium_patch_len=16, medium_stride=8,
                 slow_patch_len=32, slow_stride=16,
                 max_seq_len=1024, n_layers=3, d_model=128, n_heads=8,
                 n_heads_channel=4, d_ff=256, norm='BatchNorm',
                 attn_dropout=0., dropout=0., activation="gelu",
                 res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True,
                 fc_dropout=0., head_dropout=0, padding_patch=None,
                 individual=False, revin=True, affine=True, subtract_last=False,
                 verbose=False, use_channel_attention=True,
                 use_group_channel_attention=None,
                 use_global_channel_attention=None,
                 use_trend_residual_head=False,
                 use_inertia_residual_head=False,
                 trend_smooth_kernel=5,
                 residual_scale=0.1,
                 use_fault_class_head=False,
                 num_fault_classes=5,
                 fault_head_hidden=None,
                 fault_head_dropout=None,
                 multi_scale_layout='auto', group_permutation_seed=2026, **kwargs):
        super().__init__()

        if use_trend_residual_head and use_inertia_residual_head:
            raise ValueError("use_trend_residual_head and use_inertia_residual_head are mutually exclusive.")

        if multi_scale_layout == 'auto':
            multi_scale_layout = 'pvcable' if c_in == 13 else 'generic'
        if multi_scale_layout not in {'pvcable', 'generic', 'random_fixed'}:
            raise ValueError("multi_scale_layout must be one of: auto, pvcable, generic, random_fixed.")
        if multi_scale_layout in {'pvcable', 'random_fixed'} and c_in != 13:
            raise ValueError("PVCable and random_fixed multi-scale layouts expect the 13-channel PVCable feature layout.")

        self.c_in = c_in
        self.d_model = d_model
        self.revin = revin
        self.padding_patch = padding_patch
        self.target_idx = c_in - 1
        self.multi_scale_layout = multi_scale_layout
        self.use_trend_residual_head = use_trend_residual_head
        self.use_inertia_residual_head = use_inertia_residual_head
        self.use_fault_class_head = use_fault_class_head
        self.use_group_channel_attention = (
            use_channel_attention if use_group_channel_attention is None
            else use_group_channel_attention
        )
        self.use_global_channel_attention = (
            use_channel_attention if use_global_channel_attention is None
            else use_global_channel_attention
        )

        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Feature layout for PVCable forecast:
        # 0 tension_endA_std_kN, 1 tension_mid_std_kN, 2 tension_endB_std_kN,
        # 3 vibration_mid_mean_mm_s, 4 vibration_mid_std_mm_s,
        # 5 deflection_mid_mean_mm, 6 deflection_mid_std_mm,
        # 7 temperature_C, 8 humidity_percent, 9 wind_speed_m_s,
        # 10 wind_x, 11 wind_y, 12 tension_mid_mean_kN.
        # short_response: local structural fluctuation variables
        # environmental: external wind/humidity excitation variables
        # trend: smooth state/trend variables, including the target channel
        if self.multi_scale_layout == 'pvcable':
            self.group_configs = [
                ("short_response", [0, 1, 2, 4, 6]),
                ("environmental", [8, 9, 10, 11]),
                ("trend", [3, 5, 7, 12]),
            ]
        elif self.multi_scale_layout == 'random_fixed':
            shuffled = np.random.default_rng(group_permutation_seed).permutation(12).tolist()
            self.group_configs = [
                ("short_response", shuffled[:5]),
                ("environmental", shuffled[5:9]),
                ("trend", shuffled[9:] + [12]),
            ]
            print(f"Random-fixed PVCable groups (seed={group_permutation_seed}): {self.group_configs}")
        else:
            self.group_configs = self._build_generic_group_configs(c_in)

        patch_params = {
            "short_response": (fast_patch_len, fast_stride),
            "environmental": (medium_patch_len, medium_stride),
            "trend": (slow_patch_len, slow_stride),
        }
        self.scale_configs = [
            (name, indices, *patch_params[name])
            for name, indices in self.group_configs
        ]

        self.padding_layers = nn.ModuleDict()
        self.encoders = nn.ModuleDict()
        self.patch_nums = {}

        for name, indices in self.group_configs:
            self.register_buffer(f"{name}_indices", torch.tensor(indices, dtype=torch.long), persistent=False)

        for name, indices, patch_len, stride in self.scale_configs:
            patch_num = int((context_window - patch_len) / stride + 1)
            if padding_patch == 'end':
                self.padding_layers[name] = nn.ReplicationPad1d((0, stride))
                patch_num += 1

            self.patch_nums[name] = patch_num
            self.encoders[name] = CTEncoder(
                c_in=len(indices), patch_num=patch_num, patch_len=patch_len,
                max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                n_heads=n_heads, n_heads_channel=n_heads_channel, d_ff=d_ff,
                attn_dropout=attn_dropout, dropout=dropout, activation=activation,
                norm=norm, res_attention=res_attention, pre_norm=pre_norm,
                pe=pe, learn_pe=learn_pe,
                use_channel_attention=self.use_group_channel_attention,
                **kwargs
            )

        self.target_patch_num = max(self.patch_nums.values())
        if self.use_global_channel_attention:
            self.global_channel_attn = PatchwiseChannelAttention(
                d_model, n_heads=n_heads_channel,
                attn_dropout=attn_dropout, proj_dropout=dropout
            )
            self.global_channel_norm = nn.LayerNorm(d_model)
            self.global_channel_dropout = nn.Dropout(dropout)

        self.head_nf = d_model * self.target_patch_num
        if self.use_inertia_residual_head:
            self.head = InertiaResidualFusionHead(
                d_model, self.target_patch_num, target_window,
                head_dropout=head_dropout,
            )
        elif self.use_trend_residual_head:
            self.head = TrendResidualFusionHead(
                d_model, self.target_patch_num, target_window,
                head_dropout=head_dropout,
                trend_smooth_kernel=trend_smooth_kernel,
                residual_scale=residual_scale,
            )
        else:
            self.head = TargetFusionHead(d_model, self.target_patch_num, target_window, head_dropout=head_dropout)

        if self.use_fault_class_head:
            self.fault_head = FaultClassificationHead(
                d_model=d_model,
                num_classes=num_fault_classes,
                hidden_dim=fault_head_hidden,
                head_dropout=head_dropout if fault_head_dropout is None else fault_head_dropout,
            )

    def _get_indices(self, name):
        return getattr(self, f"{name}_indices")

    def _encode_group(self, z_group, encoder_key, patch_len, stride, return_attn=False):
        if self.padding_patch == 'end':
            z_group = self.padding_layers[encoder_key](z_group)

        z_group = z_group.unfold(dimension=-1, size=patch_len, step=stride)
        z_group = z_group.permute(0, 1, 3, 2)

        if return_attn:
            z_group, ch_attn = self.encoders[encoder_key](z_group, return_attn=True)
            return z_group, ch_attn

        return self.encoders[encoder_key](z_group), None

    def _align_patch_axis(self, z):
        """z: [B, C, D, P] -> [B, C, D, target_patch_num]"""
        if z.shape[-1] == self.target_patch_num:
            return z
        B, C, D, P = z.shape
        z = z.reshape(B * C, D, P)
        z = F.interpolate(z, size=self.target_patch_num, mode='linear', align_corners=False)
        return z.reshape(B, C, D, self.target_patch_num)

    def _denorm_target(self, z):
        """Denormalize target-only output. z: [B, 1, pred_len]."""
        z = z.permute(0, 2, 1)  # [B, pred_len, 1]

        if self.revin_layer.affine:
            weight = self.revin_layer.affine_weight[self.target_idx].view(1, 1, 1)
            bias = self.revin_layer.affine_bias[self.target_idx].view(1, 1, 1)
            z = z - bias
            z = z / (weight + self.revin_layer.eps * self.revin_layer.eps)

        if self.revin_layer.subtract_last:
            center = self.revin_layer.last[:, :, self.target_idx:self.target_idx + 1]
        else:
            center = self.revin_layer.mean[:, :, self.target_idx:self.target_idx + 1]
        stdev = self.revin_layer.stdev[:, :, self.target_idx:self.target_idx + 1]

        z = z * stdev + center
        return z.permute(0, 2, 1)  # [B, 1, pred_len]

    def forward(self, z, return_attn=False):
        """z: [B, C, seq_len]"""
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1)
        target_history = z[:, self.target_idx, :]

        encoded = z.new_zeros(z.shape[0], self.c_in, self.d_model, self.target_patch_num)
        attn_by_scale = {}

        for name, _, patch_len, stride in self.scale_configs:
            indices = self._get_indices(name)
            z_group_base = z.index_select(dim=1, index=indices)
            z_group, ch_attn = self._encode_group(
                z_group_base, name, patch_len, stride, return_attn=return_attn
            )
            if return_attn and ch_attn is not None:
                attn_by_scale[name] = ch_attn

            z_group = self._align_patch_axis(z_group)
            encoded[:, indices, :, :] = z_group

        if self.use_global_channel_attention:
            # Fuse the restored 13-channel representation so the target channel
            # can attend to fast/medium/slow variables before the prediction head.
            encoded_for_attn = encoded.permute(0, 1, 3, 2)
            if return_attn:
                encoded_attn, global_attn = self.global_channel_attn(encoded_for_attn, return_attn=True)
                attn_by_scale["global"] = global_attn
            else:
                encoded_attn = self.global_channel_attn(encoded_for_attn)
            encoded = self.global_channel_norm(
                encoded_for_attn + self.global_channel_dropout(encoded_attn)
            ).permute(0, 1, 3, 2)

        fault_logits = self.fault_head(encoded, self.target_idx) if self.use_fault_class_head else None

        if self.use_inertia_residual_head:
            z = self.head(encoded, self.target_idx, target_history)
        else:
            z = self.head(encoded, self.target_idx)

        if self.revin:
            z = self._denorm_target(z)

        if return_attn:
            if self.use_fault_class_head:
                return z, fault_logits, attn_by_scale
            return z, attn_by_scale
        if self.use_fault_class_head:
            return z, fault_logits
        return z
