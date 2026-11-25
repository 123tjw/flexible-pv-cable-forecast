__all__ = ['MyPatchTST']

import torch.nn as nn
from typing import Optional
from torch import Tensor
from layers.MyPatchTST_backbone import CTPatchTST_backbone, MultiScaleCTPatchTST_backbone


class Model(nn.Module):
    """MyPatchTST - PatchTST with cross-channel attention.

    Simple entry point that delegates everything to CTPatchTST_backbone.
    Input:  [B, seq_len, C]
    Output: [B, pred_len, C]
    """

    def __init__(self, configs, max_seq_len: Optional[int] = 1024,
                 d_k: Optional[int] = None, d_v: Optional[int] = None,
                 norm: str = 'BatchNorm', attn_dropout: float = 0.,
                 act: str = "gelu", key_padding_mask: bool = 'auto',
                 padding_var: Optional[int] = None,
                 attn_mask: Optional[Tensor] = None,
                 res_attention: bool = False, pre_norm: bool = False,
                 store_attn: bool = False, pe: str = 'zeros',
                 learn_pe: bool = True, pretrain_head: bool = False,
                 head_type: str = 'flatten', verbose: bool = False, **kwargs):

        super().__init__()

        n_heads_channel = getattr(configs, 'n_heads_channel', 4)
        self.use_fault_class_head = getattr(configs, 'use_fault_class_head', False)

        fault_head_dropout = getattr(configs, 'fault_head_dropout', None)
        if fault_head_dropout is not None and fault_head_dropout < 0:
            fault_head_dropout = None

        common_kwargs = dict(
            c_in=configs.enc_in,
            context_window=configs.seq_len,
            target_window=configs.pred_len,
            max_seq_len=max_seq_len,
            n_layers=configs.e_layers,
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            n_heads_channel=n_heads_channel,
            d_ff=configs.d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=configs.dropout,
            activation=act,
            res_attention=res_attention,
            pre_norm=pre_norm,
            pe=pe,
            learn_pe=learn_pe,
            fc_dropout=configs.fc_dropout,
            head_dropout=configs.head_dropout,
            padding_patch=configs.padding_patch,
            individual=configs.individual,
            revin=configs.revin,
            affine=configs.affine,
            subtract_last=configs.subtract_last,
            verbose=verbose,
            use_channel_attention=getattr(configs, 'use_channel_attention', True),
            use_fault_class_head=self.use_fault_class_head,
            num_fault_classes=getattr(configs, 'num_fault_classes', getattr(configs, 'num_classes', 5)),
            fault_head_hidden=getattr(configs, 'fault_head_hidden', None) or None,
            fault_head_dropout=fault_head_dropout,
            **kwargs
        )

        if getattr(configs, 'use_multi_scale', False):
            self.model = MultiScaleCTPatchTST_backbone(
                fast_patch_len=getattr(configs, 'fast_patch_len', 8),
                fast_stride=getattr(configs, 'fast_stride', 4),
                medium_patch_len=getattr(configs, 'medium_patch_len', configs.patch_len),
                medium_stride=getattr(configs, 'medium_stride', configs.stride),
                slow_patch_len=getattr(configs, 'slow_patch_len', 32),
                slow_stride=getattr(configs, 'slow_stride', 16),
                use_group_channel_attention=getattr(
                    configs, 'use_group_channel_attention',
                    getattr(configs, 'use_channel_attention', True)
                ),
                use_global_channel_attention=getattr(
                    configs, 'use_global_channel_attention',
                    getattr(configs, 'use_channel_attention', True)
                ),
                use_trend_residual_head=getattr(configs, 'use_trend_residual_head', False),
                use_inertia_residual_head=getattr(configs, 'use_inertia_residual_head', False),
                trend_smooth_kernel=getattr(configs, 'trend_smooth_kernel', 5),
                residual_scale=getattr(configs, 'residual_scale', 0.1),
                multi_scale_layout=getattr(configs, 'multi_scale_layout', 'auto'),
                group_permutation_seed=getattr(configs, 'group_permutation_seed', 2026),
                **common_kwargs
            )
        else:
            self.model = CTPatchTST_backbone(
                patch_len=configs.patch_len,
                stride=configs.stride,
                use_single_global_channel_attention=getattr(
                    configs, 'use_global_channel_attention_only',
                    getattr(configs, 'use_single_global_channel_attention', False)
                ),
                **common_kwargs
            )

    def forward(self, x):
        # x: [B, L, C] -> backbone expects [B, C, L] -> returns [B, C, pred_len]
        x = x.permute(0, 2, 1)
        x = self.model(x)
        if isinstance(x, tuple):
            forecast, fault_logits = x[:2]
            forecast = forecast.permute(0, 2, 1)  # [B, pred_len, C]
            return forecast, fault_logits
        x = x.permute(0, 2, 1)  # [B, pred_len, C]
        return x
