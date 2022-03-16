# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple
import torch


def _xavier_init(linear):
    """
    Performs the Xavier weight initialization of the linear layer `linear`.
    """
    torch.nn.init.xavier_uniform_(linear.weight.data)




class MLPWithInputSkips(torch.nn.Module):
    """
    Implements the multi-layer perceptron architecture of the Neural Radiance Field.

    As such, `MLPWithInputSkips` is a multi layer perceptron consisting
    of a sequence of linear layers with ReLU activations.

    Additionally, for a set of predefined layers `input_skips`, the forward pass
    appends a skip tensor `z` to the output of the preceding layer.

    Note that this follows the architecture described in the Supplementary
    Material (Fig. 7) of [1].

    References:
        [1] Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik
            and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng:
            NeRF: Representing Scenes as Neural Radiance Fields for View
            Synthesis, ECCV2020
    """

    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        output_dim: int,
        skip_dim: int,
        hidden_dim: int,
        input_skips: Tuple[int] = (),
    ):
        """
        Args:
            n_layers: The number of linear layers of the MLP.
            input_dim: The number of channels of the input tensor.
            output_dim: The number of channels of the output.
            skip_dim: The number of channels of the tensor `z` appended when
                evaluating the skip layers.
            hidden_dim: The number of hidden units of the MLP.
            input_skips: The list of layer indices at which we append the skip
                tensor `z`.
        """
        super().__init__()
        layers = []
        for layeri in range(n_layers):
            if layeri == 0:
                dimin = input_dim
                dimout = hidden_dim
            elif layeri in input_skips:
                dimin = hidden_dim + skip_dim
                dimout = hidden_dim
            else:
                dimin = hidden_dim
                dimout = hidden_dim
            linear = torch.nn.Linear(dimin, dimout)
            _xavier_init(linear)
            layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))
        self.mlp = torch.nn.ModuleList(layers)
        self._input_skips = set(input_skips)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: The input tensor of shape `(..., input_dim)`.
            z: The input skip tensor of shape `(..., skip_dim)` which is appended
                to layers whose indices are specified by `input_skips`.
        Returns:
            y: The output tensor of shape `(..., output_dim)`.
        """
        y = x
        for li, layer in enumerate(self.mlp):
            if li in self._input_skips:
                y = torch.cat((y, z), dim=-1)
            y = layer(y)
        return y
