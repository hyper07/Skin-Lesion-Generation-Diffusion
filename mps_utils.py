#!/usr/bin/env python3
"""
MPS (Metal Performance Shaders) utility for Mac M1/M2/M3 chips.
This script checks MPS availability and provides helper functions for GPU acceleration.
"""

import torch
import platform
import subprocess
import sys

def check_mps_availability():
    """
    Check if MPS (Metal Performance Shaders) is available on this Mac.
    
    Returns:
        dict: Dictionary with MPS availability info
    """
    info = {
        'platform': platform.system(),
        'machine': platform.machine(),
        'python_version': sys.version,
        'pytorch_version': torch.__version__,
        'mps_available': False,
        'mps_built': False,
        'device_count': 0,
        'recommended_device': 'cpu'
    }
    
    # Check if running on macOS
    if platform.system() == 'Darwin':
        info['mps_built'] = torch.backends.mps.is_built()
        info['mps_available'] = torch.backends.mps.is_available()
        
        if info['mps_available']:
            info['device_count'] = 1  # MPS typically shows as single device
            info['recommended_device'] = 'mps'
        
        # Check for Apple Silicon
        if platform.machine() in ['arm64', 'aarch64']:
            info['apple_silicon'] = True
        else:
            info['apple_silicon'] = False
    
    return info

def get_optimal_device():
    """
    Get the optimal device for PyTorch operations.
    
    Returns:
        torch.device: The best available device
    """
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def print_device_info():
    """Print detailed information about available devices."""
    info = check_mps_availability()
    
    print("🖥️  Device Information")
    print("=" * 50)
    print(f"Platform: {info['platform']} ({info['machine']})")
    print(f"Python Version: {info['python_version'].split()[0]}")
    print(f"PyTorch Version: {info['pytorch_version']}")
    
    if info['platform'] == 'Darwin':
        print(f"Apple Silicon: {'Yes' if info.get('apple_silicon', False) else 'No'}")
        print(f"MPS Built: {'Yes' if info['mps_built'] else 'No'}")
        print(f"MPS Available: {'Yes' if info['mps_available'] else 'No'}")
        
        if info['mps_available']:
            print("🚀 Metal Performance Shaders (MPS) is available!")
            print("   You can use GPU acceleration on your Mac.")
            device = torch.device('mps')
            print(f"   Recommended device: {device}")
            
            # Test MPS functionality
            try:
                test_tensor = torch.randn(100, 100).to(device)
                result = torch.mm(test_tensor, test_tensor.T)
                print("✅ MPS functionality test: PASSED")
            except Exception as e:
                print(f"❌ MPS functionality test: FAILED - {e}")
        else:
            print("⚠️  MPS not available. Using CPU.")
            if not info['mps_built']:
                print("   MPS support was not built with this PyTorch installation.")
            print("   Recommended device: cpu")
    
    print("\n🔧 Device Usage Examples:")
    print("=" * 50)
    
    optimal_device = get_optimal_device()
    print(f"# Get optimal device")
    print(f"device = torch.device('{optimal_device}')")
    print()
    print("# Move tensors to device")
    print("tensor = torch.randn(3, 224, 224).to(device)")
    print("model = MyModel().to(device)")
    print()
    print("# Training loop example")
    print("for batch in dataloader:")
    print("    inputs, labels = batch")
    print("    inputs = inputs.to(device)")
    print("    labels = labels.to(device)")
    print("    # ... rest of training code")

def benchmark_devices():
    """
    Benchmark different devices for matrix multiplication.
    """
    print("\n⚡ Performance Benchmark")
    print("=" * 50)
    
    # Test parameters
    size = 2048
    iterations = 10
    
    devices_to_test = ['cpu']
    if torch.backends.mps.is_available():
        devices_to_test.append('mps')
    if torch.cuda.is_available():
        devices_to_test.append('cuda')
    
    print(f"Matrix size: {size}x{size}")
    print(f"Iterations: {iterations}")
    print()
    
    results = {}
    
    for device_name in devices_to_test:
        device = torch.device(device_name)
        print(f"Testing {device_name.upper()}...")
        
        try:
            # Create tensors
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            
            # Warm up
            for _ in range(3):
                _ = torch.mm(a, b)
            
            if device_name == 'mps':
                torch.mps.synchronize()
            elif device_name == 'cuda':
                torch.cuda.synchronize()
            
            # Benchmark
            import time
            start_time = time.time()
            
            for _ in range(iterations):
                result = torch.mm(a, b)
                if device_name == 'mps':
                    torch.mps.synchronize()
                elif device_name == 'cuda':
                    torch.cuda.synchronize()
            
            end_time = time.time()
            avg_time = (end_time - start_time) / iterations
            results[device_name] = avg_time
            
            print(f"  Average time: {avg_time:.4f} seconds")
            
        except Exception as e:
            print(f"  Error: {e}")
            results[device_name] = float('inf')
    
    # Show results
    print("\n📊 Results Summary:")
    if len(results) > 1:
        fastest_device = min(results, key=results.get)
        print(f"Fastest device: {fastest_device.upper()}")
        
        if 'cpu' in results:
            for device, time_taken in results.items():
                if device != 'cpu' and time_taken != float('inf'):
                    speedup = results['cpu'] / time_taken
                    print(f"{device.upper()} is {speedup:.2f}x faster than CPU")

def create_mps_config():
    """Create a configuration dictionary for MPS usage."""
    return {
        'device': get_optimal_device(),
        'use_mps': torch.backends.mps.is_available(),
        'mixed_precision': torch.backends.mps.is_available(),  # MPS supports mixed precision
        'dataloader_num_workers': 0 if torch.backends.mps.is_available() else 4,  # MPS works better with num_workers=0
    }

def main():
    """Main function to run device checks and benchmarks."""
    print_device_info()
    
    # Ask user if they want to run benchmark
    if torch.backends.mps.is_available():
        response = input("\nWould you like to run a performance benchmark? (y/n): ").lower()
        if response in ['y', 'yes']:
            benchmark_devices()
    
    print("\n🛠️  MPS Configuration for Training:")
    print("=" * 50)
    config = create_mps_config()
    for key, value in config.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()