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


class GridField(nn.Module):
    """Multi-resolution feature grids with a small per-pixel head, for a 2-D map.

    Dense version of the hash encoding of neural_wfa (``HashMLP``, version 1;
    Mueller et al. 2022): each level is a learnable ``(F, ry, rx)`` grid, the
    levels go geometrically from ``base_res`` to ``max_res``, and a map of size
    (ny, nx) <= 512 needs no hashing. Because the field is only evaluated on
    the pixel grid, the bilinear lookup is a bilinear upsampling of each level
    (``F.interpolate``, align_corners=True, same as sampling the [-1, 1]
    coordinates of :func:`to_coordinates`), and the head is a stack of 1x1
    convolutions. Cost per evaluation is ~0.1-0.3 s on a 512x512 map (CPU),
    against ~10 s for :class:`MultiMLP` with m_freqs=512.

    ``groups``: list of ``(output_channels, max_res)``; each group has its own
    grids and head, so a group with a lower ``max_res`` is spatially smoother
    (the role of the per-output ``sigma`` of MultiMLP).
    ``zero_init``: output channels whose last-layer weights and bias start at
    zero, so they output exactly 0 at the start (the ``init`` values of
    :class:`~neural_me.core.atmospheres.nf_heads.ThermoHead`). Do not zero the
    transverse-field channels: Q and U are quadratic in them, so at exactly 0
    their gradient vanishes and they never move.
    """

    def __init__(self, grid_shape: Tuple[int, int], dim_out: int = 8,
                 groups: Optional[List[Tuple[Tuple[int, ...], int]]] = None,
                 n_levels: int = 12, base_res: int = 4, features_per_level: int = 4,
                 dim_hidden: int = 32, num_layers: int = 2,
                 activation: nn.Module = nn.GELU(), init_scale: float = 1e-4,
                 zero_init: Tuple[int, ...] = ()):
        super().__init__()
        if len(grid_shape) != 2:
            raise ValueError(f"GridField needs a 2-D grid, got {grid_shape}")
        self.grid_shape = tuple(int(s) for s in grid_shape)
        self.dim_out = dim_out
        if groups is None:
            groups = [(tuple(range(dim_out)), max(self.grid_shape))]
        covered = sorted(c for ch, _ in groups for c in ch)
        if covered != list(range(dim_out)):
            raise ValueError(f"groups must cover channels 0..{dim_out - 1} once, got {covered}")
        self.groups = [(tuple(ch), int(mr)) for ch, mr in groups]
        self.levels = nn.ModuleList()
        self.heads = nn.ModuleList()
        for _, max_res in self.groups:
            res = self.resolutions(base_res, max_res, n_levels)
            grids = nn.ParameterList([
                nn.Parameter(torch.empty(1, features_per_level, ry, rx).uniform_(-init_scale, init_scale))
                for ry, rx in res])
            self.levels.append(grids)
        for ch, _ in self.groups:
            dim_in = n_levels * features_per_level
            layers = [nn.Conv2d(dim_in, dim_hidden, 1), activation]
            for _ in range(num_layers - 1):
                layers += [nn.Conv2d(dim_hidden, dim_hidden, 1), activation]
            last = nn.Conv2d(dim_hidden, len(ch), 1)
            with torch.no_grad():
                for k, c in enumerate(ch):
                    if c in zero_init:
                        last.weight[k].zero_()
                        last.bias[k].zero_()
            self.heads.append(nn.Sequential(*layers, last))

    def resolutions(self, base_res: int, max_res: int, n_levels: int) -> List[Tuple[int, int]]:
        """(ry, rx) per level: geometric in [base_res, max_res], capped at the map size."""
        ny, nx = self.grid_shape
        b = (max_res / base_res) ** (1.0 / max(n_levels - 1, 1))
        out = []
        for i in range(n_levels):
            r = base_res * b ** i
            out.append((max(2, min(ny, int(round(r)))), max(2, min(nx, int(round(r))))))
        return out

    def forward(self) -> torch.Tensor:
        """(ny * nx, dim_out) outputs in the row-major order of :func:`to_coordinates`."""
        ny, nx = self.grid_shape
        out = [None] * self.dim_out
        for (ch, _), grids, head in zip(self.groups, self.levels, self.heads):
            feat = torch.cat([nn.functional.interpolate(g, size=(ny, nx), mode="bilinear", align_corners=True)
                              for g in grids], dim=1)
            y = head(feat)[0]                         # (len(ch), ny, nx)
            for k, c in enumerate(ch):
                out[c] = y[k].reshape(-1)
        return torch.stack(out, dim=-1)
