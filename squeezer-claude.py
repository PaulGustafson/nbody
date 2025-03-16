import triton
import triton.language as tl
import torch

DEVICE = torch.device('cuda:0')

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
    tl.store(dx_ptr + i_offsets[:, None] * N + j_offsets[None, :], dx, mask=i_mask[:, None] & j_mask[None, :])

def compute_dx(pos_x):
    N = pos_x.shape[0]
    dx = torch.empty((N, N), device=pos_x.device, dtype=pos_x.dtype)
    BLOCK_SIZE = 32
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE, (N + BLOCK_SIZE - 1) // BLOCK_SIZE)
    pairwise_dx_kernel[grid](
        pos_x,
        dx,
        N,
        BLOCK_SIZE
    )
    return dx

def compute_dx_torch(pos_x):
    return pos_x.unsqueeze(0) - pos_x.unsqueeze(1)

def verify_dx(N, rtol=1e-3):
    torch.manual_seed(0)
    pos_x = torch.randn((N, 1), device=DEVICE, dtype=torch.float32)
    dx_triton = compute_dx(pos_x)
    dx_torch = compute_dx_torch(pos_x)
    print(dx_triton)
    print(dx_torch)
    assert torch.allclose(dx_triton, dx_torch, atol=rtol, rtol=rtol)
    print("✅ Triton and Torch match")


verify_dx(2)