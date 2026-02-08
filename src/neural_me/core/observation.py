
import torch
import numpy as np
from typing import Optional, Union, List, Tuple

class Observation:
    """
    Container for spectropolarimetric observation data.
    
    Handles:
    1. Storage of Stokes profiles (I, Q, U, V).
    2. Spectral line information (wavelengths).
    3. Masking of active spectral regions.
    4. Auto-flattening of spatial dimensions for batch processing.
    """
    
    def __init__(
        self,
        data: Union[torch.Tensor, np.ndarray],
        wavelengths: Union[torch.Tensor, np.ndarray],
        active_wav_idx: Optional[Union[List[int], torch.Tensor]] = None,
        device: str = "cpu"
    ):
        """
        Args:
            data: Shape (..., 4, n_lambda). Expected to be [I, Q, U, V].
            wavelengths: Shape (n_lambda,). Angstroms relative to line center.
            active_wav_idx: Indices of wavelengths to use for inversion.
            device: 'cpu' or 'cuda'.
        """
        # Convert to Tensor and Device
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        if isinstance(wavelengths, np.ndarray):
            wavelengths = torch.from_numpy(wavelengths).float()
            
        self.device = torch.device(device)
        self.data = data.to(self.device)
        self.wavelengths = wavelengths.to(self.device)
        
        # Determine spatial/temporal dimensions
        # Data shape is (..., 4, n_lambda)
        self.grid_shape = self.data.shape[:-2] 
        self.n_stokes = self.data.shape[-2]
        self.n_lambda = self.data.shape[-1]
        
        # Flatten spatial dimensions for easier processing (N_pixels, 4, n_lambda)
        self.flat_data = self.data.reshape(-1, self.n_stokes, self.n_lambda)
        self.n_pixels = self.flat_data.shape[0]
        
        # Handle Active Indexs
        if active_wav_idx is not None:
             if not isinstance(active_wav_idx, torch.Tensor):
                 self.active_wav_idx = torch.tensor(active_wav_idx, device=self.device)
             else:
                 self.active_wav_idx = active_wav_idx.to(self.device)
        else:
            self.active_wav_idx = torch.arange(self.n_lambda, device=self.device)
            
    @property
    def stokes_I(self):
        return self.flat_data[:, 0, :]
        
    @property
    def stokes_Q(self):
        return self.flat_data[:, 1, :]
        
    @property
    def stokes_U(self):
        return self.flat_data[:, 2, :]
        
    @property
    def stokes_V(self):
        return self.flat_data[:, 3, :]
        
    @property
    def active_wavelengths(self):
        return self.wavelengths[self.active_wav_idx]
    
    def to(self, device):
        """Move data to device."""
        # Convert active_wav_idx to list if it exists, or keep as tensor?
        # Constructor expects list or tensor.
        return Observation(
            self.data.to(device),
            self.wavelengths.to(device),
            self.active_wav_idx.to(device) if self.active_wav_idx is not None else None,
            device
        )

    def get_coordinates(self) -> torch.Tensor:
        """
        Generates normalized coordinates in range [-1, 1] for each pixel.
        Returns tensor of shape (N_pixels, D).
        """
        dims = len(self.grid_shape)
        if dims == 0: # 1 pixel case?
             return torch.zeros((1, 1), device=self.device)
             
        coords_list = []
        for d in range(dims):
            dim_size = self.grid_shape[d]
            if dim_size > 1:
                c = torch.linspace(-1, 1, dim_size, device=self.device)
            else:
                c = torch.zeros(1, device=self.device)
            coords_list.append(c)
            
        mesh = torch.meshgrid(*coords_list, indexing='ij')
        stacked = torch.stack(mesh, dim=-1)
        return stacked.reshape(-1, dims)
