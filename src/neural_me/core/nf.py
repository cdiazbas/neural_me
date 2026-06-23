"""
Neural Field (NF) core modules.
"""

import math
from typing import Tuple, List, Union, Optional
import numpy as np
import torch
import torch.nn as nn


def to_coordinates(grid_shape: Tuple[int, ...], device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Generate normalized coordinates in [-1, 1] for the given grid shape.
    
    Args:
        grid_shape: Tuple of spatial dimensions, e.g., (Ny, Nx).
        device: Target PyTorch device.
        
    Returns:
        Tensor of shape (prod(grid_shape), len(grid_shape)) containing coordinates.
    """
    grids = [torch.linspace(-1.0, 1.0, steps=s, device=device) for s in grid_shape]
    grid_tensors = torch.meshgrid(*grids, indexing="ij")
    coords = torch.stack([g.flatten() for g in grid_tensors], dim=-1)
    return coords


class MLP(nn.Module):
    """
    Multilayer Perceptron (MLP) with optional ResNet connections and Random Fourier Features.
    """
    def __init__(
        self,
        dim_in: int = 2,
        dim_out: int = 1,
        num_resnet_blocks: int = 3,
        num_layers_per_block: int = 2,
        dim_hidden: int = 50,
        activation: nn.Module = nn.GELU(),
        fourier_features: bool = False,
        m_freqs: int = 100,
        sigma: Union[float, List[float]] = 10.0,
        tune_beta: bool = False,
    ):
        super().__init__()
        self.num_resnet_blocks = num_resnet_blocks
        self.num_layers_per_block = num_layers_per_block
        self.fourier_features = fourier_features
        self.activation = activation
        self.tune_beta = tune_beta
        self.sigma = sigma
        num_neurons = dim_hidden

        if tune_beta:
            self.beta0 = nn.Parameter(torch.ones(1, 1))
            self.beta = nn.Parameter(torch.ones(self.num_resnet_blocks, self.num_layers_per_block))
        else:
            self.register_buffer("beta0", torch.ones(1, 1))
            self.register_buffer("beta", torch.ones(self.num_resnet_blocks, self.num_layers_per_block))

        if fourier_features:
            n_param = len(sigma) if isinstance(sigma, list) else dim_in
            B = torch.from_numpy(np.random.normal(0.0, sigma, (m_freqs, n_param)).T).float()
            self.register_buffer("B", B)
            self.first = nn.Linear(2 * m_freqs + dim_in, num_neurons)
        else:
            self.first = nn.Linear(dim_in, num_neurons)

        self.resblocks = nn.ModuleList([
            nn.ModuleList([nn.Linear(num_neurons, num_neurons) for _ in range(num_layers_per_block)])
            for _ in range(num_resnet_blocks)
        ])
        self.last = nn.Linear(num_neurons, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xx = x.clone()

        if self.fourier_features:
            # Matmul expects x: (N, dim_in) and B: (dim_in, m_freqs)
            proj = torch.matmul(x, self.B)
            cosx = torch.cos(proj)
            sinx = torch.sin(proj)
            x = torch.cat((cosx, sinx, xx), dim=-1)

        x = self.activation(self.beta0 * self.first(x))

        for i in range(self.num_resnet_blocks):
            z = self.activation(self.beta[i][0] * self.resblocks[i][0](x))
            for j in range(1, self.num_layers_per_block):
                z = self.activation(self.beta[i][j] * self.resblocks[i][j](z))
            x = z + x

        out = self.last(x)
        return out


class MultiMLP(nn.Module):
    """
    Ensemble model where a separate MLP is allocated for each parameter prediction,
    supporting independent frequency scaling (sigma) per parameter.
    """
    def __init__(
        self,
        dim_in: int = 2,
        dim_out: int = 1,
        num_resnet_blocks: int = 3,
        num_layers_per_block: int = 2,
        dim_hidden: int = 50,
        activation: nn.Module = nn.GELU(),
        fourier_features: bool = False,
        m_freqs: int = 100,
        sigma: Union[List[float], List[List[float]]] = 10.0,
        tune_beta: bool = False,
    ):
        super().__init__()
        
        # If sigma is a single value/list, replicate it for all outputs
        if not isinstance(sigma, list):
            sigma_list = [sigma] * dim_out
        elif len(sigma) != dim_out:
            sigma_list = [sigma] * dim_out
        else:
            sigma_list = sigma

        self.multi_net = nn.ModuleList()
        for i in range(dim_out):
            net = MLP(
                dim_in=dim_in,
                dim_out=1,
                num_resnet_blocks=num_resnet_blocks,
                num_layers_per_block=num_layers_per_block,
                dim_hidden=dim_hidden,
                activation=activation,
                fourier_features=fourier_features,
                m_freqs=m_freqs,
                sigma=sigma_list[i],
                tune_beta=tune_beta
            )
            self.multi_net.append(net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Evaluate each network and stack outputs
        outs = [net(x).squeeze(dim=-1) for net in self.multi_net]
        return torch.stack(outs, dim=-1)
