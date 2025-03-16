import torch
import triton
import triton.language as tl

# Print versions to verify installation
print(f"Triton version: {triton.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")

DEVICE = torch.device('cuda:0')
