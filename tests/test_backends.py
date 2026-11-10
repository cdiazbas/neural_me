import torch
import numpy as np
import pytest
import sys
import os

# Ensure we can import neural_me
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from neural_me import Observation, MEProblem

@pytest.mark.cpp
def test_pytorch_vs_cpp_backend():
    # Setup a mock observation
    wavelengths = torch.linspace(-0.5, 0.5, 20)
    # Shape: (1, 1, 4, 20)
    data = torch.zeros((1, 1, 4, 20))
    obs = Observation(data=data, wavelengths=wavelengths)

    # Initialize problems for both backends
    prob_torch = MEProblem(observation=obs, line=6302, backend="torch", precision="float64")
    prob_cpp = MEProblem(observation=obs, line=6302, backend="cpp", precision="float64")

    # Physical parameters stack: [B, gamma, xi, vlos, eta0, a, ddop, S0, S1]
    params = torch.tensor([[1000.0, 1.5, 0.8, 1.2, 80.0, 0.15, 0.04, 0.6, 0.4]], dtype=torch.float64)

    # Compute forward models
    I_t, Q_t, U_t, V_t = prob_torch.model(params, wavelengths.unsqueeze(0))
    I_c, Q_c, U_c, V_c = prob_cpp.model(params, wavelengths.unsqueeze(0))

    # Verify they match (Voigt approximations differ slightly between Python and C++)
    np.testing.assert_allclose(I_t.detach().numpy(), I_c.detach().numpy(), rtol=5e-3, atol=5e-3)
    np.testing.assert_allclose(Q_t.detach().numpy(), Q_c.detach().numpy(), rtol=5e-3, atol=5e-3)
    np.testing.assert_allclose(U_t.detach().numpy(), U_c.detach().numpy(), rtol=5e-3, atol=5e-3)
    np.testing.assert_allclose(V_t.detach().numpy(), V_c.detach().numpy(), rtol=5e-3, atol=5e-3)
    print("Test passed: PyTorch and C++ backends are fully consistent!")

if __name__ == "__main__":
    test_pytorch_vs_cpp_backend()
