from typing import Optional, Any

import torch
import torch.functional as F
from torch import Tensor


class SelfAttention(torch.nn.Module):
    r"""SelfAttention is made a TransformerEncoderLayer with no feedforward network.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        src = torch.rand(10, 32, 512)
        out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead=8, dropout=0.0, activation="relu", bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None):
        super(SelfAttention, self).__init__()
        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias, add_bias_kv=add_bias_kv,
                                                     add_zero_attn=add_zero_attn, kdim=kdim, vdim=vdim)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.norm1 = torch.nn.LayerNorm(d_model)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = torch.nn.ReLU(inplace=True)
        super(SelfAttention, self).__setstate__(state)

    def forward(self, src: Tensor) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
        Shape:
            see the docs in Transformer class.
        """
        src = src.permute(1,0,2)
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.activation(src)
        return src.permute(1,0,2)

def _get_activation_fn(activation):
    if activation == "relu":
        return torch.nn.ReLU(inplace=True)
    elif activation == "gelu":
        return torch.nn.GELU()