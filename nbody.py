import torch
import triton
import triton.language as tl

DEVICE = torch.device('cuda:0')


@triton.jit
def nbody_force_kernel(
    pos_x_ptr: tl.tensor,
    pos_y_ptr: tl.tensor,
    mass_ptr: tl.tensor, 
    force_x_ptr: tl.tensor,
    force_y_ptr: tl.tensor,
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
    y = tl.load(pos_y_ptr + offsets, mask=mask)
    m = tl.load(mass_ptr + offsets, mask=mask)

    # Initialize force accumulators
    fx = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    fy = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Loop over all particles
    for j in range(0, n_elements, BLOCK_SIZE):
        j_idx = j + tl.arange(0, BLOCK_SIZE)
        j_mask = j_idx < n_elements
        
        # Load other particles
        xj = tl.load(pos_x_ptr + j_idx, mask=j_mask)
        yj = tl.load(pos_y_ptr + j_idx, mask=j_mask)
        mj = tl.load(mass_ptr + j_idx, mask=j_mask)

        # Compute forces for each pair
        for k in range(BLOCK_SIZE):
            #if j_mask[k]:
            dx = xj - x
            dy = yj - y
            r_sq = dx * dx + dy * dy + softening
            r_inv = 1.0 / tl.sqrt(r_sq)
            f = G * m * mj * r_inv * r_inv * r_inv
            
            fx = fx + tl.where(mask, f * dx, 0.0)
            fy = fy + tl.where(mask, f * dy, 0.0)

    # Store results
    tl.store(force_x_ptr + offsets, fx, mask=mask)
    tl.store(force_y_ptr + offsets, fy, mask=mask)

def compute_forces(pos_x, pos_y, mass, G=1.0, softening=1e-6):
    # Ensure inputs are float32 and on GPU
    pos_x = pos_x.contiguous().to(torch.float32)
    pos_y = pos_y.contiguous().to(torch.float32)
    mass = mass.contiguous().to(torch.float32)
    
    N = pos_x.shape[0]
    force_x = torch.empty_like(pos_x, dtype=torch.float32, device=DEVICE)
    force_y = torch.empty_like(pos_y, dtype=torch.float32, device=DEVICE)

    assert pos_x.device == DEVICE and pos_y.device == DEVICE \
        and mass.device == DEVICE and force_x.device == DEVICE \
            and force_y.device == DEVICE
    
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']), )

    nbody_force_kernel[grid](
        pos_x_ptr=pos_x,
        pos_y_ptr=pos_y,
        mass_ptr=mass,
        force_x_ptr=force_x,
        force_y_ptr=force_y,
        n_elements=N,
        G=G,
        softening=softening,
        BLOCK_SIZE=1024,
    )
    return force_x, force_y

# Test the implementation
N = 98432
pos_x = torch.rand(N, device=DEVICE) * 10.0
pos_y = torch.rand(N, device=DEVICE) * 10.0
mass = torch.ones(N, device=DEVICE)

# Compute forces
force_x, force_y = compute_forces(pos_x, pos_y, mass)

print(f"Input shape: {pos_x.shape}")
print(f"Output shape: {force_x.shape}")
print(f"Force magnitude range: {force_x.min():.3f} to {force_x.max():.3f}")

# Verify forces are reasonable
total_force = force_x.sum(dim=0) + force_y.sum(dim=0)
print(f"Total force (should be close to zero): ({total_force.item():.3e})")