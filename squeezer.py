import triton
import triton.language as tl
import torch

@triton.jit
def pairwise_dx_kernel(pos_x_ptr, dx_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid_x = tl.program_id(0)  # Row index (i)
    pid_y = tl.program_id(1)  # Column index (j)

    # Compute offsets for this block
    i_offsets = pid_x * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    j_offsets = pid_y * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    i_mask = i_offsets < N
    j_mask = j_offsets < N

    # Load positions
    xi = tl.load(pos_x_ptr + i_offsets, mask=i_mask, other=0.0)
    xj = tl.load(pos_x_ptr + j_offsets, mask=j_mask, other=0.0)

    # Compute pairwise difference (broadcasting)
    dx = xi[:, None] - xj[None, :]

    # Store result
    tl.store(dx_ptr + i_offsets * N + j_offsets, dx, mask=i_mask[:, None] & j_mask[None, :])

# Launcher
def compute_dx(pos_x):
    N = pos_x.shape[0]
    dx = torch.zeros(N, N, device='cuda', dtype=torch.float32)
    BLOCK_SIZE = 32
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE, (N + BLOCK_SIZE - 1) // BLOCK_SIZE)
    pairwise_dx_kernel[grid](
        pos_x,
        dx,
        N,
        BLOCK_SIZE
    )
    return dx

# Test
N = 4
pos_x = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda', dtype=torch.float32)
dx = compute_dx(pos_x)
print(dx)
# Compare with PyTorch
print(pos_x.unsqueeze(0) - pos_x.unsqueeze(1))
