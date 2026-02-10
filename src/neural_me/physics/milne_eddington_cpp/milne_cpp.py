import torch
from torch.autograd import Variable
from . import MilneEddington as ME
import numpy as np
from einops import rearrange

class MILNE_function(torch.autograd.Function):
    """
    PyTorch autograd function wrapping the C++ Milne-Eddington synthesis.
    
    Parameter Ordering Note:
        The C++ backend expects model parameters in this order:
        [B, gamma, xi, vlos, ddop, eta0, a, S0, S1]
        
        The forward() signature uses a different order for PyTorch API compatibility:
        (B, gamma, xi, vlos, eta0, a, ddop, S0, S1)
        
        The model array construction (line ~34) and gradient extraction (backward)
        are carefully ordered to maintain consistency. This is intentional.
    
    Normalization Notes:
        - eta0 is scaled by 1/sqrt(pi) in forward() to match PyTorch profile normalization
        - grad_eta0 is also scaled by 1/sqrt(pi) in backward() (chain rule)
    
    C++ Backend Limitations:
        - vlos is hardcoded clipped to [-13, +13] km/s in the C++ library
        - gamma at 0 or π: response function gradients become zero, causing optimization to get stuck.
          We add a small epsilon to avoid these exact boundary values.
          PyTorch autodiff does not have this limitation.
        - Response functions may differ from PyTorch autodiff at boundary cases
    """

    @staticmethod
    def forward(ctx, B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1, milne_instance, mu):

        # print('==> forward')
        # Move tensors to CPU before numpy conversion (C++ backend is CPU-only)
        # Get precision from the milne instance
        precision = milne_instance._get_dtype()
        np_dtype = np.float32 if precision == 'float32' else np.float64
        
        # Pre-allocate model array to avoid redundant copies/transpositions
        # Shape: (1, n_pixels, 9) to match MilneEddington.py expectations
        # Clamp gamma to avoid exactly 0 or pi where C++ gradients become zero
        eps = 1e-6
        n_pixels = B.shape[0]
        model = np.zeros((1, n_pixels, 9), dtype=np_dtype)
        
        # Fill model array efficiently
        model[0, :, 0] = B.detach().cpu().numpy()[:, 0]
        model[0, :, 1] = np.clip(gamma.detach().cpu().numpy()[:, 0], eps, np.pi - eps) # Inc
        
        # Azimuth wrapping
        np_xi = xi.detach().cpu().numpy()[:, 0]
        np_xi[np_xi < 0] += np.pi
        np_xi[np_xi > np.pi] -= np.pi
        model[0, :, 2] = np_xi
        
        model[0, :, 3] = vlos.detach().cpu().numpy()[:, 0]
        model[0, :, 4] = ddop.detach().cpu().numpy()[:, 0]
        
        # Normalize eta0 and apply damping positivity
        model[0, :, 5] = eta0.detach().cpu().numpy()[:, 0] / np.sqrt(np.pi)
        model[0, :, 6] = np.abs(a.detach().cpu().numpy()[:, 0] + eps)
        
        model[0, :, 7] = S_0.detach().cpu().numpy()[:, 0]
        model[0, :, 8] = S_1.detach().cpu().numpy()[:, 0]

        if isinstance(mu, torch.Tensor):
            np_mu = mu.detach().cpu().numpy().astype(np_dtype)
        else:
            np_mu = float(mu)
            
        nbatch = n_pixels

        # Use synthesize_rf which returns (stokes, rf)
        # Note: synthesize_rf in MilneEddington.py uses _get_dtype() to ensure arrays are contiguous and correct type
        stokes, dstokes = milne_instance.synthesize_rf(model, mu=np_mu)
        stokes = rearrange(stokes, 'nx ny ns nw -> (ny nx) (ns nw)')
        
        # Zero-copy reshape: (nx, ny, 9, 4, nw) -> (Pixels, 9, TotalWvl)
        # This is a view in NumPy, avoiding a 500MB+ copy for 256x256 grid.
        dstokes = dstokes.reshape(n_pixels, 9, -1)

        ctx.dstokes = dstokes
        ctx.nbatch = nbatch
        ctx.nl4 = stokes.shape[0]*stokes.shape[1]
        ctx.input_device = B.device  # Store original device for backward


        # global RF_global
        # RF_global = ctx.dstokes
            
        return torch.as_tensor(stokes, dtype=B.dtype, device=B.device)


        # return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        # print('==> backward')

        # Get the device from context (stored during forward)
        device = ctx.input_device
        
        # Convert response functions to tensors on the correct device (Zero-copy if on CPU)
        # ctx.dstokes layout: (Pixels, 9, TotalWvl)
        # grad_output layout: (Pixels, TotalWvl)
        dstokes_t = torch.as_tensor(ctx.dstokes, device=device)
        
        # Compute all 9 gradients at once using batched matrix multiplication
        # (Pixels, 9, TotalWvl) @ (Pixels, TotalWvl, 1) -> (Pixels, 9, 1)
        grad_all = torch.bmm(dstokes_t, grad_output.unsqueeze(-1)).squeeze(-1)
        
        # Unpack gradients in forward signature order:
        # CPP order: B[0], gamma[1], xi[2], vlos[3], ddop[4], eta0[5], a[6], S0[7], S1[8]
        # Signature: B, gamma, xi, vlos, eta0, a, ddop, S0, S1
        grad_B = grad_all[:, 0][:, None]
        grad_gamma = grad_all[:, 1][:, None]
        grad_xi = grad_all[:, 2][:, None]
        grad_vlos = grad_all[:, 3][:, None]
        
        # eta0 gradient needs sqrt(pi) scaling (it was 6th in CPP: index 5)
        grad_eta0 = grad_all[:, 5][:, None] / np.sqrt(np.pi)
        
        grad_a = grad_all[:, 6][:, None]
        grad_ddop = grad_all[:, 4][:, None]
        grad_S0 = grad_all[:, 7][:, None]
        grad_S1 = grad_all[:, 8][:, None]
        
        return grad_B, grad_gamma, grad_xi, grad_vlos, grad_eta0, grad_a, grad_ddop, grad_S0, grad_S1, None, None












def findgrid(w, dw, extra = 5):
    """
    Findgrid creates a regular wavelength grid 
    with a step of dw that includes all points in 
    input array w. It adds extra points at the edges
    for convolution purposes

    Returns the new array and the positions of the
    wavelengths points from w in the new array
    """
    nw = np.int32(np.rint(w/dw))
    print('nw',nw)
    nnw = nw[-1] - nw[0] + 1 + 2*extra
    
    iw = np.arange(nnw, dtype='float64')*dw - extra*dw + w[0]

    idx = np.arange(w.size, dtype='int32')
    for ii in range(w.size):
        idx[ii] = np.argmin(np.abs(iw-w[ii]))

    return iw, idx



class MILNE(torch.nn.Module):
    """
    Adding the possibility of using a spectral PSF
    """
    def __init__(self, lambda0, JUp, JLow, gUp, gLow, lambda_axis, mu, spectralPSF=None, nthreads=None, precision='float64'):
        super(MILNE, self).__init__()

        self.spectralPSF = spectralPSF

        if isinstance(lambda_axis, torch.Tensor):
            lambda_axis = lambda_axis.clone().detach().numpy()


        if spectralPSF is None:
            iw1 = lambda_axis + lambda0
            regions = [[iw1, None]]
            lines = [int(lambda0)]
        
        else:
            # Assume that the spectralPSF is sampled in a efficient way
            wl_psf = self.spectralPSF[:,0]
            tr_psf = self.spectralPSF[:,1]
            dw = wl_psf[1] - wl_psf[0]
            new_lambda_axis, idx = findgrid(lambda_axis, dw/1e3)
            print('lambda_axis',lambda_axis)
            print('new_lambda_axis',new_lambda_axis)
            
            # self.idx is for Stokes I, but I have concatenated all the Stokes, so I need to make the 
            # new index for all the Stokes: IQUV
            print('idx',idx)
            self.idx = []
            for i in range(4):
                self.idx.extend(idx + i*new_lambda_axis.size)
            print('self.idx',self.idx)
            
            
            print('[INFO] Using spectral PSF. Grid size: ',lambda_axis.size,'->',new_lambda_axis.size)
            
            # Create the new regions
            iw1 = new_lambda_axis + lambda0
            # regions = [[iw1, None]]#tr_psf/tr_psf.sum()]]
            regions = [[iw1, tr_psf/tr_psf.sum()]]
            # regions = [[lambda_axis + lambda0, None]]
            lines = [int(lambda0)]
            
        
        if nthreads is None:
            nthreads = torch.get_num_threads()
        print('Using num_threads: ',nthreads)
        print('Using precision: ', precision)
        self.milne_class = ME.MilneEddington(regions, lines, nthreads=nthreads, precision=precision)
        self.mu = mu

    def forward(self,  B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1):
        if self.spectralPSF is None:
            self.stokes = MILNE_function.apply(B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1, self.milne_class, self.mu)
            
        else:
            # Use only the true spectral points for comparison
            self.stokes = MILNE_function.apply(B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1, self.milne_class, self.mu)[:,self.idx]
            # print('self.stokes.shape',self.stokes.shape)
            # [:,self.idx]
        
        return self.stokes




if (__name__ == '__main__'):
    import torch.autograd.functional as af
    import numpy as np
    
    lambda0 = 6301.5080
    JUp = 2.0
    JLow = 2.0
    gUp = 1.5
    gLow = 1.833
    lambdaStart = 6300.8
    lambdaStep = 0.03
    nLambda = 50
    lambda_axis = np.linspace(lambdaStart,lambdaStep*nLambda +lambdaStart,nLambda)


    B = torch.tensor([[992.]], requires_grad=True)
    gamma = torch.tensor([[np.deg2rad(134.)]], requires_grad=True)
    xi = torch.tensor([[np.deg2rad(145.)]], requires_grad=True)
    vlos = torch.tensor([[1.0]], requires_grad=True)
    eta0 = torch.tensor([[73.]], requires_grad=True)
    a = torch.tensor([[0.2*10]], requires_grad=True)
    ddop = torch.tensor([[0.02]], requires_grad=True)
    S_0 = torch.tensor([[0.5]], requires_grad=True)
    S_1 = torch.tensor([[0.5]], requires_grad=True)
    mu = torch.tensor([1.0])


    fn = MILNE(lambda0, JUp, JLow, gUp, gLow, lambda_axis, mu)
    stokes = fn(B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1)
    print('stokes.shape',stokes.shape)


    # (B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1)
    # 0,  1,     2,  3,    5,    6, 4,    7,   8
    loss = torch.mean(stokes**2)
    loss.backward()   


    print(RF_global.shape)
    # S_1
    with torch.no_grad():
        S_1new = torch.tensor([[S_1.data]], requires_grad=False)
        eps = S_1new.data*0.001
        S_1new += eps
        print(S_1,'->',S_1new)
        stokes2 = fn(B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1new)
        rf_num = ((stokes2 - stokes) / eps).detach().numpy()
    
        loss2 = torch.mean(stokes2**2)

        rf_num2 = ((loss2 - loss) / eps).detach().numpy()
        
        print('(loss2 - loss)',(loss2 - loss))    
        print('new.grad',S_1.grad.detach().numpy())
        print('rf_num2',rf_num2)
        

        
    # print(rf_num.shape)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(211)
    # plt.plot(stokes[0,:].detach().numpy())
    plt.plot(stokes[0,:].detach().numpy()-stokes2[0,:].detach().numpy(),ls='--')
    plt.subplot(212)
    plt.plot(rf_num[0,:])
    plt.plot(RF_global[0,:,8],ls='--')
    plt.show()


