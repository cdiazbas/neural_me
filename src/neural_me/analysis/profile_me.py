import torch
import torch.profiler
import os
import sys

# Ensure we can import neural_me
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from neural_me.physics.milne_eddington import MilneEddington


def profile_me_model(batch_size=4096, n_lambda=128, steps=10):
    """
    Profiles the ME model execution.
    """
    print(
        f"Profiling ME model with batch_size={batch_size}, n_lambda={n_lambda}, steps={steps}"
    )

    # Init model
    nlinea = 4
    model = MilneEddington(line=nlinea)

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
    print("Warming up (compiling JIT if applicable)...")
    for _ in range(5):
        with torch.no_grad():
            _ = model(params, wl)

    print("Starting profiler...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=steps, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            "./log/me_model_profile"
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(1 + 1 + steps):
            with torch.no_grad():
                _ = model(params, wl)
            prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    print("Trace saved to ./log/me_model_profile")


if __name__ == "__main__":
    profile_me_model()
