#!/usr/bin/env python3
"""
Test GPU Acceleration for openWakeWord

This script demonstrates GPU acceleration using OpenCL on the RK3588's Mali-G610 GPU
for the melspectrogram and embedding models.

Requirements:
- openWakeWord with GPU support
- ONNX Runtime with OpenCL support
- Mali-G610 GPU drivers

Usage:
    python test_gpu_acceleration.py [--framework FRAMEWORK] [--device DEVICE]
"""

import argparse
import numpy as np
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from openwakeword import Model
    from openwakeword.utils import AudioFeatures
except ImportError as e:
    print(f"Error importing openWakeWord: {e}")
    print("Please ensure openWakeWord is installed.")
    sys.exit(1)

def test_gpu_acceleration(framework="opencl", device="gpu"):
    """Test GPU acceleration with different frameworks."""
    
    print(f"Testing {framework} framework with device={device}")
    print("=" * 50)
    
    # Test 1: AudioFeatures (melspectrogram + embedding models)
    print("\n1. Testing AudioFeatures GPU acceleration...")
    
    try:
        # Initialize with GPU acceleration
        audio_features = AudioFeatures(
            inference_framework=framework,
            device=device,
            ncpu=1
        )
        
        # Create test audio data (1 second of random audio)
        test_audio = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
        
        # Warm up
        print("Warming up...")
        for _ in range(3):
            _ = audio_features._get_melspectrogram(test_audio)
            _ = audio_features._get_embeddings(test_audio)
        
        # Benchmark melspectrogram
        print("Benchmarking melspectrogram...")
        num_runs = 10
        start_time = time.time()
        for _ in range(num_runs):
            melspec = audio_features._get_melspectrogram(test_audio)
        melspec_time = (time.time() - start_time) / num_runs
        
        print(f"Melspectrogram time: {melspec_time*1000:.2f}ms per inference")
        print(f"Melspectrogram shape: {melspec.shape}")
        
        # Benchmark embeddings
        print("Benchmarking embeddings...")
        start_time = time.time()
        for _ in range(num_runs):
            embeddings = audio_features._get_embeddings(test_audio)
        embedding_time = (time.time() - start_time) / num_runs
        
        print(f"Embedding time: {embedding_time*1000:.2f}ms per inference")
        print(f"Embedding shape: {embeddings.shape}")
        
        # Test batch processing
        print("Testing batch processing...")
        batch_audio = np.random.randint(-32768, 32767, (5, 16000), dtype=np.int16)
        
        start_time = time.time()
        batch_melspecs = audio_features._get_melspectrogram_batch(batch_audio, batch_size=5)
        batch_melspec_time = time.time() - start_time
        
        print(f"Batch melspectrogram time: {batch_melspec_time*1000:.2f}ms for 5 samples")
        print(f"Batch melspectrogram shape: {batch_melspecs.shape}")
        
        return True
        
    except Exception as e:
        print(f"Error testing AudioFeatures: {e}")
        return False

def test_wake_word_detection(framework="opencl"):
    """Test wake word detection with GPU acceleration."""
    
    print(f"\n2. Testing wake word detection with {framework}...")
    
    try:
        # Initialize model with GPU acceleration
        model = Model(
            inference_framework=framework,
            enable_speex_noise_suppression=False,
            vad_threshold=0.0
        )
        
        # Create test audio data
        test_audio = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
        
        # Warm up
        print("Warming up...")
        for _ in range(3):
            _ = model.predict(test_audio)
        
        # Benchmark
        print("Benchmarking wake word detection...")
        num_runs = 10
        start_time = time.time()
        for _ in range(num_runs):
            predictions = model.predict(test_audio)
        inference_time = (time.time() - start_time) / num_runs
        
        print(f"Inference time: {inference_time*1000:.2f}ms per inference")
        print(f"Predictions: {predictions}")
        
        return True
        
    except Exception as e:
        print(f"Error testing wake word detection: {e}")
        return False

def compare_frameworks():
    """Compare performance between different frameworks."""
    
    print("\n3. Comparing framework performance...")
    print("=" * 50)
    
    frameworks = ["cpu", "opencl"]
    results = {}
    
    for framework in frameworks:
        print(f"\nTesting {framework} framework...")
        
        try:
            # Test AudioFeatures
            audio_features = AudioFeatures(
                inference_framework=framework,
                device="gpu" if framework == "opencl" else "cpu",
                ncpu=1
            )
            
            test_audio = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
            
            # Warm up
            for _ in range(3):
                _ = audio_features._get_melspectrogram(test_audio)
                _ = audio_features._get_embeddings(test_audio)
            
            # Benchmark
            num_runs = 20
            start_time = time.time()
            for _ in range(num_runs):
                melspec = audio_features._get_melspectrogram(test_audio)
            melspec_time = (time.time() - start_time) / num_runs
            
            start_time = time.time()
            for _ in range(num_runs):
                embeddings = audio_features._get_embeddings(test_audio)
            embedding_time = (time.time() - start_time) / num_runs
            
            results[framework] = {
                'melspec_time': melspec_time * 1000,
                'embedding_time': embedding_time * 1000,
                'total_time': (melspec_time + embedding_time) * 1000
            }
            
            print(f"  Melspectrogram: {melspec_time*1000:.2f}ms")
            print(f"  Embedding: {embedding_time*1000:.2f}ms")
            print(f"  Total: {(melspec_time + embedding_time)*1000:.2f}ms")
            
        except Exception as e:
            print(f"  Error: {e}")
            results[framework] = None
    
    # Print comparison
    print("\nPerformance Comparison:")
    print("-" * 30)
    
    if results.get("cpu") and results.get("opencl"):
        cpu_total = results["cpu"]["total_time"]
        gpu_total = results["opencl"]["total_time"]
        speedup = cpu_total / gpu_total
        
        print(f"CPU Total Time: {cpu_total:.2f}ms")
        print(f"GPU Total Time: {gpu_total:.2f}ms")
        print(f"Speedup: {speedup:.2f}x")
        
        if speedup > 1.0:
            print("✅ GPU acceleration is working!")
        else:
            print("⚠️  GPU acceleration may not be optimal")
    else:
        print("❌ Could not complete comparison")

def main():
    parser = argparse.ArgumentParser(description="Test GPU acceleration for openWakeWord")
    parser.add_argument("--framework", default="opencl", 
                       choices=["cpu", "opencl", "onnx"],
                       help="Inference framework to test")
    parser.add_argument("--device", default="gpu",
                       choices=["cpu", "gpu"],
                       help="Device to use")
    parser.add_argument("--compare", action="store_true",
                       help="Compare performance between frameworks")
    
    args = parser.parse_args()
    
    print("openWakeWord GPU Acceleration Test")
    print("=" * 50)
    
    # Check if OpenCL is available
    if args.framework == "opencl":
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if "OpenCLExecutionProvider" in providers:
                print("✅ OpenCL execution provider is available")
            else:
                print("⚠️  OpenCL execution provider not available")
                print(f"Available providers: {providers}")
        except ImportError:
            print("❌ ONNX Runtime not available")
            return
    
    # Run tests
    success1 = test_gpu_acceleration(args.framework, args.device)
    success2 = test_wake_word_detection(args.framework)
    
    if args.compare:
        compare_frameworks()
    
    if success1 and success2:
        print("\n✅ All tests completed successfully!")
    else:
        print("\n❌ Some tests failed")

if __name__ == "__main__":
    main() 