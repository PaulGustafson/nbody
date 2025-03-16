import torch
import triton
import triton.language as tl
import time
import numpy as np

DEVICE = torch.device('cuda:0')


def compute_forces_torch(pos_x, mass, G=1.0, softening=1e-6):
    # Compute pairwise distance vectors
    dx = pos_x.unsqueeze(0) - pos_x.unsqueeze(1)  # NxN matrix
    
    # Compute inverse distances cubed 
    r_sq = dx * dx  + softening
    r_inv_cube = 1.0 / torch.sqrt(r_sq) ** 3
    
    # Compute forces
    mass_matrix = mass.unsqueeze(0) * mass.unsqueeze(1)  # NxN matrix
    fx = G * (mass_matrix * r_inv_cube * dx).sum(dim=1)
    
    return fx


@triton.jit
def nbody_force_kernel(
    pos_x_ptr: tl.tensor,
    mass_ptr: tl.tensor, 
    force_x_ptr: tl.tensor,
    n_elements: tl.int32,
    G: tl.float32,
    softening: tl.float32,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    # Get the program ID
    pid = tl.program_id(axis=0)
    
    # Compute indices for this block
    block_start = pid * BLOCK_SIZE 
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load position and mass for this particle block
    # Use explicit pointer arithmetic without array indexing
    x = tl.load(pos_x_ptr + offsets, mask=mask)
    m = tl.load(mass_ptr + offsets, mask=mask)

    # Initialize force accumulators
    fx = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Loop over all particles
    for j in range(0, n_elements, BLOCK_SIZE):
        #tl.device_print('j', j)
        j_idx = j + tl.arange(0, BLOCK_SIZE)
        j_mask = j_idx < n_elements
        
        # Load other particles
        xj = tl.load(pos_x_ptr + j_idx, mask=j_mask)
        mj = tl.load(mass_ptr + j_idx, mask=j_mask)

        # Compute forces for each pair
        #dx = xj[:, None] - x[None, :] 
        dx = tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32)
         dx[i,j] = xj[i] - x[j]



        r_sq = dx * dx + softening
        r_inv = 1.0 / tl.sqrt(r_sq)
        f = G * m * mj * r_inv * r_inv * r_inv
    
        fx = fx + tl.sum(tl.where(j_mask, f * dx, 0.0), axis=0)

    # Store results
    tl.store(force_x_ptr + offsets, fx, mask=mask)

def compute_forces(pos_x, mass, G=1.0, softening=1e-6):
    # Ensure inputs are float32 and on GPU
    pos_x = pos_x.contiguous().to(torch.float32)
    mass = mass.contiguous().to(torch.float32)
    
    N = pos_x.shape[0]
    force_x = torch.empty_like(pos_x, dtype=torch.float32, device=DEVICE)

    assert pos_x.device == DEVICE and mass.device == DEVICE \
        and force_x.device == DEVICE
    
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']), )

    nbody_force_kernel[grid](
        pos_x_ptr=pos_x,
        mass_ptr=mass,
        force_x_ptr=force_x,
        n_elements=N,
        G=G,
        softening=softening,
        BLOCK_SIZE=1024,
    )
    return force_x


def compute_forces_torch(pos_x, mass, G=1.0, softening=1e-6):
    """Reference implementation using PyTorch for verification"""
    # Ensure inputs are float32 and on GPU
    pos_x = pos_x.contiguous().to(torch.float32)
    mass = mass.contiguous().to(torch.float32)
    
    # Compute pairwise distances
    dx = pos_x.unsqueeze(0) - pos_x.unsqueeze(1)  # (N,N)
    
    # Compute forces
    r_sq = dx * dx + softening
    r_inv = 1.0 / torch.sqrt(r_sq)
    m1m2 = mass.unsqueeze(0) * mass.unsqueeze(1)  # (N,N) 
    f = G * m1m2 * r_inv * r_inv * r_inv
    
    # Compute force components
    fx = (f * dx).sum(dim=1)  # Sum forces from all particles
    
    return fx

def verify_forces(N, rtol=1e-3):
    """Verify Triton kernel against PyTorch reference implementation"""
    # Generate random test data
    torch.manual_seed(0)
        # pos_x = torch.rand(N, device=DEVICE)
        # pos_y = torch.rand(N, device=DEVICE)
        # mass = torch.rand(N, device=DEVICE)

    pos_x = torch.tensor([0.0, 1.0], device=DEVICE)
    mass = torch.tensor([1.0, 1.0], device=DEVICE)

    # Run both implementations
    fx_tri = compute_forces(pos_x, mass)
    fx_ref = compute_forces_torch(pos_x, mass)

    print(fx_tri, fx_ref)
    
    # Compare results with both rtol and atol specified
    # torch.testing.assert_close(fx_tri, fx_ref, rtol=rtol, atol=rtol)
    # print(f"✓ Triton kernel matches PyTorch reference (N={N}, rtol={rtol})")

if __name__ == "__main__":
    # Verify for different sizes
    for N in [32, 128, 1024]:
        verify_forces(N)
