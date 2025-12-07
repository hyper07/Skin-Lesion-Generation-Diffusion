"""
Unified device detection and information utility
Automatically detects and reports CUDA (NVIDIA), MPS (Apple Silicon), or CPU availability
"""

import sys
import platform
import subprocess
import torch


def check_nvidia_gpu():
    """Check if NVIDIA GPU is available via nvidia-smi"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 2:
                        name = parts[0].strip()
                        memory = parts[1].strip()
                        gpus.append({'name': name, 'memory': memory})
            return True, gpus
        return False, []
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return False, []


def check_cuda():
    """Check if CUDA is available in PyTorch"""
    if not torch.cuda.is_available():
        return False, None
    
    device_count = torch.cuda.device_count()
    devices = []
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        devices.append({
            'index': i,
            'name': props.name,
            'memory_total': f"{props.total_memory / 1024**3:.2f} GB",
            'compute_capability': f"{props.major}.{props.minor}"
        })
    return True, devices


def check_mps():
    """Check if MPS (Metal Performance Shaders) is available"""
    if not hasattr(torch.backends, 'mps'):
        return False, None
    
    if not torch.backends.mps.is_available():
        return False, None
    
    if not torch.backends.mps.is_built():
        return False, None
    
    # Get system info
    try:
        import psutil
        memory = psutil.virtual_memory()
        total_memory = f"{memory.total / 1024**3:.2f} GB"
    except ImportError:
        total_memory = "Unknown"
    
    return True, {
        'available': True,
        'built': True,
        'total_memory': total_memory,
        'platform': platform.system(),
        'processor': platform.processor()
    }


def detect_device():
    """Automatically detect the best available device"""
    # Check CUDA first (NVIDIA GPU)
    cuda_available, cuda_devices = check_cuda()
    if cuda_available:
        return 'cuda', cuda_devices[0] if cuda_devices else None
    
    # Check MPS (Apple Silicon)
    mps_available, mps_info = check_mps()
    if mps_available:
        return 'mps', mps_info
    
    # Fallback to CPU
    return 'cpu', None


def print_device_info():
    """Print comprehensive device information"""
    print("=" * 70)
    print("Device Detection and Information")
    print("=" * 70)
    print()
    
    # System Information
    print("System Information:")
    print(f"  Platform: {platform.system()} {platform.release()}")
    print(f"  Processor: {platform.processor()}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    print()
    
    # Check NVIDIA GPU (via nvidia-smi)
    print("NVIDIA GPU Detection (nvidia-smi):")
    nvidia_available, nvidia_gpus = check_nvidia_gpu()
    if nvidia_available and nvidia_gpus:
        for i, gpu in enumerate(nvidia_gpus):
            print(f"  GPU {i}: {gpu['name']}")
            print(f"    Memory: {gpu['memory']}")
    else:
        print("  No NVIDIA GPUs detected via nvidia-smi")
    print()
    
    # Check CUDA (PyTorch)
    print("CUDA (PyTorch):")
    cuda_available, cuda_devices = check_cuda()
    if cuda_available and cuda_devices:
        print(f"  ✓ CUDA is available")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  cuDNN Version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'Not available'}")
        print(f"  Number of GPUs: {len(cuda_devices)}")
        for device in cuda_devices:
            print(f"    GPU {device['index']}: {device['name']}")
            print(f"      Memory: {device['memory_total']}")
            print(f"      Compute Capability: {device['compute_capability']}")
    else:
        print("  ✗ CUDA is not available in PyTorch")
    print()
    
    # Check MPS (Apple Silicon)
    print("MPS (Apple Silicon):")
    mps_available, mps_info = check_mps()
    if mps_available:
        print("  ✓ MPS is available")
        if isinstance(mps_info, dict):
            print(f"    Total Memory: {mps_info.get('total_memory', 'Unknown')}")
            print(f"    Platform: {mps_info.get('platform', 'Unknown')}")
    else:
        print("  ✗ MPS is not available")
    print()
    
    # CPU Information
    print("CPU:")
    try:
        import psutil
        cpu_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        print(f"  Cores (logical): {cpu_count}")
        if cpu_freq:
            print(f"  Frequency: {cpu_freq.current:.2f} MHz (max: {cpu_freq.max:.2f} MHz)")
    except ImportError:
        print("  (Install psutil for detailed CPU information)")
    print()
    
    # Recommended Device
    device_type, device_info = detect_device()
    print("=" * 70)
    print(f"Recommended Device: {device_type.upper()}")
    if device_type == 'cuda' and device_info:
        print(f"  Using: {device_info['name']}")
    elif device_type == 'mps':
        print("  Using: Apple Silicon GPU (MPS)")
    elif device_type == 'cpu':
        print("  Using: CPU (no GPU acceleration available)")
    print("=" * 70)
    print()
    
    return device_type, device_info


def get_device():
    """Get the best available device (same as used in training scripts)"""
    # Temporarily force CPU to avoid MPS locking issues
    return torch.device("cpu")
    # if torch.cuda.is_available():
    #     return torch.device("cuda")
    # if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    #     return torch.device("mps")
    # return torch.device("cpu")


def main():
    """Main function to run device detection"""
    try:
        device_type, device_info = print_device_info()
        
        # Test device
        print("Testing device with a simple tensor operation...")
        device = get_device()
        print(f"  Device: {device}")
        
        try:
            x = torch.randn(100, 100).to(device)
            y = torch.randn(100, 100).to(device)
            z = torch.matmul(x, y)
            print(f"  ✓ Successfully created and multiplied tensors on {device}")
            print(f"  Result shape: {z.shape}")
        except Exception as e:
            print(f"  ✗ Error testing device: {e}")
        
        print()
        print("Device detection complete!")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during device detection: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

