import torch
from torch.autograd import Variable
from . import MilneEddington as ME
import numpy as np
from einops import rearrange

class MILNE_function(torch.autograd.Function):

    @staticmethod
    def forward(ctx, B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1, milne_instance, mu):

        # print('==> forward')
        np_B = B.detach().numpy()
        np_gamma = gamma.detach().numpy()
        np_xi = xi.detach().numpy()
        np_vlos = vlos.detach().numpy()
        np_a = a.detach().numpy()
        np_S_0 = S_0.detach().numpy()
        np_S_1 = S_1.detach().numpy()
        np_ddop = ddop.detach().numpy()
        np_eta0 = eta0.detach().numpy()
        if isinstance(mu, torch.Tensor):
            np_mu = mu.detach().numpy().astype(np.float64)
        else:
            np_mu = float(mu)

        np_xi[np_xi < 0] += (np.pi)
        np_xi[np_xi > (torch.pi)] -= (np.pi)

        model = np.asarray([np_B, np_gamma, np_xi, np_vlos, np_ddop, np_eta0, np_a, np_S_0, np_S_1]).T
        nbatch = model.shape[-1]

        stokes, dstokes = milne_instance.synthesize_rf(model.astype(np.float64),mu=mu)
        stokes = rearrange(stokes, 'nx ny ns nw -> (ny nx) (ns nw)')
        dstokes = rearrange(dstokes, 'nx ny np ns nw -> (ny nx) (ns nw) np')

        ctx.dstokes = dstokes
        ctx.nbatch = nbatch
        ctx.nl4 = stokes.shape[0]*stokes.shape[1]


        global RF_global
        RF_global = ctx.dstokes
            
        return torch.as_tensor(stokes, dtype=B.dtype)


        # return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        # print('==> backward')

        # grad_input[input < 0] = 0
        # 0, 1, 2, 3, 5, 6, 4, 7, 8
        return torch.as_tensor(ctx.dstokes[:,:,0])*grad_output, torch.as_tensor(ctx.dstokes[:,:,1])*grad_output, torch.as_tensor(ctx.dstokes[:,:,2])*grad_output, torch.as_tensor(ctx.dstokes[:,:,3])*grad_output, torch.as_tensor(ctx.dstokes[:,:,5])*grad_output, torch.as_tensor(ctx.dstokes[:,:,6])*grad_output, torch.as_tensor(ctx.dstokes[:,:,4])*grad_output, torch.as_tensor(ctx.dstokes[:,:,7])*grad_output, torch.as_tensor(ctx.dstokes[:,:,8])*grad_output, None, None












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
    def __init__(self, lambda0, JUp, JLow, gUp, gLow, lambda_axis, mu, spectralPSF=None, nthreads=None):
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
        self.milne_class = ME.MilneEddington(regions, lines, nthreads=nthreads)
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


