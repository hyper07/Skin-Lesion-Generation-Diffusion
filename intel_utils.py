#!/usr/bin/env python3
"""
Intel PC GPU/CPU utility for performance optimization.
This script checks CUDA availability and provides helper functions for Intel systems.
"""

import torch
import platform
import subprocess
import sys
import os
import psutil

def check_intel_system():
    """
    Check if running on Intel system and gather system information.
    
    Returns:
        dict: Dictionary with Intel system info
    """
    info = {
        'platform': platform.system(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'python_version': sys.version,
        'pytorch_version': torch.__version__,
        'is_intel': False,
        'cpu_count': psutil.cpu_count(),
        'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'cuda_available': False,
        'cuda_version': None,
        'gpu_count': 0,
        'gpu_names': [],
        'recommended_device': 'cpu'
    }
    
    # Check if Intel processor
    processor_info = platform.processor().lower()
    if 'intel' in processor_info or 'genuine intel' in processor_info:
        info['is_intel'] = True
    
    # Check CUDA availability
    if torch.cuda.is_available():
        info['cuda_available'] = True
        info['cuda_version'] = torch.version.cuda
        info['gpu_count'] = torch.cuda.device_count()
        info['recommended_device'] = 'cuda'
        
        # Get GPU names
        for i in range(info['gpu_count']):
            gpu_name = torch.cuda.get_device_name(i)
            info['gpu_names'].append(gpu_name)
    
    return info

def check_nvidia_driver():
    """Check NVIDIA driver installation."""
    try:
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, 
                              text=True, 
                              timeout=10)
        return result.returncode == 0, result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, "nvidia-smi not found"

def get_optimal_device_intel():
    """
    Get the optimal device for PyTorch operations on Intel systems.
    
    Returns:
        torch.device: The best available device
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def print_intel_system_info():
    """Print detailed information about Intel system and available devices."""
    info = check_intel_system()
    
    print("💻 Intel System Information")
    print("=" * 50)
    print(f"Platform: {info['platform']} ({info['machine']})")
    print(f"Processor: {info['processor']}")
    print(f"Intel System: {'Yes' if info['is_intel'] else 'No (AMD/Other)'}")
    print(f"CPU Cores: {info['cpu_count']}")
    print(f"Memory: {info['memory_gb']} GB")
    print(f"Python Version: {info['python_version'].split()[0]}")
    print(f"PyTorch Version: {info['pytorch_version']}")
    
    # CUDA Information
    print(f"\n🎮 GPU Information")
    print("=" * 50)
    print(f"CUDA Available: {'Yes' if info['cuda_available'] else 'No'}")
    
    if info['cuda_available']:
        print(f"CUDA Version: {info['cuda_version']}")
        print(f"GPU Count: {info['gpu_count']}")
        
        for i, gpu_name in enumerate(info['gpu_names']):
            print(f"GPU {i}: {gpu_name}")
            
            # Get GPU memory info
            try:
                memory_total = torch.cuda.get_device_properties(i).total_memory
                memory_gb = memory_total / (1024**3)
                print(f"  Memory: {memory_gb:.1f} GB")
            except Exception as e:
                print(f"  Memory: Unable to detect ({e})")
        
        print("✅ NVIDIA GPU acceleration available!")
        print(f"   Recommended device: cuda")
        
        # Test CUDA functionality
        try:
            device = torch.device('cuda')
            test_tensor = torch.randn(100, 100).to(device)
            result = torch.mm(test_tensor, test_tensor.T)
            print("✅ CUDA functionality test: PASSED")
        except Exception as e:
            print(f"❌ CUDA functionality test: FAILED - {e}")
            
    else:
        print("⚠️  No CUDA GPU detected. Using CPU.")
        print("   For GPU acceleration, install NVIDIA GPU + drivers")
        print("   Recommended device: cpu")
        
        # Check if NVIDIA driver is installed
        nvidia_available, nvidia_output = check_nvidia_driver()
        if nvidia_available:
            print("✅ NVIDIA drivers detected")
            print("   Issue might be PyTorch CUDA installation")
        else:
            print("❌ NVIDIA drivers not found")
            print("   Install NVIDIA GPU drivers first")
    
    print(f"\n🔧 Optimization Recommendations")
    print("=" * 50)
    
    if info['cuda_available']:
        print("For NVIDIA GPU training:")
        print("- Use batch sizes: 64-256 (depending on VRAM)")
        print("- Enable mixed precision: torch.cuda.amp")
        print("- Set num_workers: 4-8 (match CPU cores)")
        print("- Enable pin_memory=True")
    else:
        print("For CPU training:")
        print("- Use smaller batch sizes: 16-32")
        print("- Set num_workers: 2-4")
        print("- Consider model quantization for inference")
        print("- Enable Intel MKL optimizations")
    
    print(f"\n📊 Performance Settings")
    print("=" * 50)
    config = create_intel_config()
    for key, value in config.items():
        print(f"{key}: {value}")

def benchmark_intel_devices():
    """
    Benchmark different devices for matrix multiplication on Intel systems.
    """
    print("\n⚡ Intel System Performance Benchmark")
    print("=" * 50)
    
    # Test parameters
    size = 2048
    iterations = 10
    
    devices_to_test = ['cpu']
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
            
            if device_name == 'cuda':
                torch.cuda.synchronize()
            
            # Benchmark
            import time
            start_time = time.time()
            
            for _ in range(iterations):
                result = torch.mm(a, b)
                if device_name == 'cuda':
                    torch.cuda.synchronize()
            
            end_time = time.time()
            avg_time = (end_time - start_time) / iterations
            results[device_name] = avg_time
            
            print(f"  Average time: {avg_time:.4f} seconds")
            
            # Memory usage for GPU
            if device_name == 'cuda':
                memory_used = torch.cuda.memory_allocated() / (1024**3)
                memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"  GPU Memory: {memory_used:.2f}/{memory_total:.1f} GB")
            
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

def create_intel_config():
    """Create a configuration dictionary for Intel PC usage."""
    info = check_intel_system()
    
    config = {
        'device': get_optimal_device_intel(),
        'use_cuda': torch.cuda.is_available(),
        'mixed_precision': torch.cuda.is_available(),
        'cpu_cores': info['cpu_count'],
        'memory_gb': info['memory_gb'],
    }
    
    # Optimize settings based on available hardware
    if config['use_cuda']:
        config['dataloader_num_workers'] = min(8, info['cpu_count'])
        config['pin_memory'] = True
        config['batch_size_recommendation'] = '64-256 (depending on VRAM)'
    else:
        config['dataloader_num_workers'] = min(4, info['cpu_count'] // 2)
        config['pin_memory'] = False
        config['batch_size_recommendation'] = '16-32 (for CPU training)'
    
    return config

def print_cuda_installation_guide():
    """Print CUDA installation guide for Intel PCs."""
    print("\n🚀 CUDA Installation Guide for Intel PC")
    print("=" * 50)
    
    print("1. Install NVIDIA GPU Drivers:")
    print("   - Download from: https://www.nvidia.com/drivers/")
    print("   - Or use GeForce Experience for gaming GPUs")
    print()
    
    print("2. Install CUDA Toolkit (Optional for PyTorch):")
    print("   - Download from: https://developer.nvidia.com/cuda-downloads")
    print("   - PyTorch includes its own CUDA runtime")
    print()
    
    print("3. Install PyTorch with CUDA:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print()
    
    print("4. Verify Installation:")
    print("   python -c \"import torch; print(torch.cuda.is_available())\"")
    print("   nvidia-smi")

def main():
    """Main function to run Intel system checks and benchmarks."""
    print_intel_system_info()
    
    # Ask user if they want to run benchmark
    if torch.cuda.is_available():
        response = input("\nWould you like to run a performance benchmark? (y/n): ").lower()
        if response in ['y', 'yes']:
            benchmark_intel_devices()
    else:
        response = input("\nWould you like to see CUDA installation guide? (y/n): ").lower()
        if response in ['y', 'yes']:
            print_cuda_installation_guide()

if __name__ == "__main__":
    main()