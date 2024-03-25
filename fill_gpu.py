import torch
import time

# Ensure PyTorch is using GPUs
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. This test requires GPUs.")

# Get the number of GPUs available
num_gpus = torch.cuda.device_count()
print(f"{num_gpus} GPU(s) detected.")

# List to hold tensors
tensors = []

# Approximate VRAM usage per GPU (in GB), adjust based on your requirements
vram_gb_per_gpu = 75

# Loop over every GPU and allocate a tensor
for gpu_index in range(num_gpus):
    device = torch.device(f"cuda:{gpu_index}")
    gpu_name = torch.cuda.get_device_name(gpu_index)
    print(f"Allocating tensor on {gpu_name} (Device {gpu_index}).")

    # Calculate the number of elements to allocate based on desired VRAM usage
    bytes_per_gb = 1024 ** 3
    dtype_size = 4  # float32 is 4 bytes
    num_elements = (vram_gb_per_gpu * bytes_per_gb) // dtype_size

    # Create a large tensor of ones on this GPU
    large_tensor = torch.ones(num_elements, device=device, dtype=torch.float32)

    # Perform a simple operation that requires computation
    large_tensor *= 2

    # Add the tensor to the list to prevent it from being freed
    tensors.append(large_tensor)

print("Tensors allocated in VRAM on all GPUs. You can now check the VRAM usage with nvidia-smi.")
print("The script will run indefinitely. Use Ctrl+C to stop and release VRAM.")

try:
    while True:
        # Keep the script running to hold the tensors in VRAM
        time.sleep(1)  # Sleep to prevent this loop from consuming CPU unnecessarily
except KeyboardInterrupt:
    print("Interrupt received, releasing VRAM and exiting...")
