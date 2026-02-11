import torch
import time
import os
import sys

# Ensure we can import neural_me
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from neural_me.physics.milne_eddington import MilneEddington


def benchmark_me_model(batch_size=4096, n_lambda=128, steps=100, warmup=10):
    """
    Benchmarks the ME model execution speed.
    """
    print(
        f"Benchmarking ME model with batch_size={batch_size}, n_lambda={n_lambda}, steps={steps}"
    )

    # Init model
    nlinea = 4
    model = MilneEddington(nlinea=nlinea)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # Generate random inputs
    torch.manual_seed(42)
    B = torch.rand(batch_size, device=device) * 3.0
    gamma = torch.rand(batch_size, device=device) * 3.14159
    xi = torch.rand(batch_size, device=device) * 3.14159
    vlos = (torch.rand(batch_size, device=device) - 0.5) * 10.0
    eta0 = torch.rand(batch_size, device=device) * 10.0 + 1.0
    a = torch.rand(batch_size, device=device) * 1.0
    ddop = torch.rand(batch_size, device=device) * 0.05 + 0.02
    S_0 = torch.rand(batch_size, device=device)
    S_1 = torch.rand(batch_size, device=device)

    params = torch.stack([B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1], dim=1)
    wl = (
        torch.linspace(6300.5, 6302.5, n_lambda, device=device)
        .unsqueeze(0)
        .repeat(batch_size, 1)
    )

    # Warmup
    print(f"Warming up ({warmup} steps)...")
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(params, wl)

    if device.type == "cuda":
        torch.cuda.synchronize()

    print("Starting benchmark...")
    start_time = time.time()

    for _ in range(steps):
        with torch.no_grad():
            _ = model(params, wl)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / steps
    throughput = (batch_size * steps) / total_time

    print(f"Total time: {total_time:.4f} s")
    print(f"Average time per batch: {avg_time * 1000:.4f} ms")
    print(f"Throughput: {throughput:.2f} samples/s")


if __name__ == "__main__":
    benchmark_me_model()
