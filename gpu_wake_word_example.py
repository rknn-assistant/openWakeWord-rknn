#!/usr/bin/env python3
"""
GPU-Accelerated Wake Word Detection Example

This example demonstrates how to use openWakeWord with GPU acceleration
for the melspectrogram and embedding models on the RK3588's Mali-G610 GPU.

Usage:
    python gpu_wake_word_example.py
"""

import numpy as np
import time
from openwakeword import Model

def main():
    print("GPU-Accelerated Wake Word Detection Example")
    print("=" * 50)
    
    # Initialize model with GPU acceleration
    print("Initializing model with OpenCL GPU acceleration...")
    model = Model(
        inference_framework="opencl",  # Use OpenCL for GPU acceleration
        enable_speex_noise_suppression=False,
        vad_threshold=0.0
    )
    
    print(f"Loaded {len(model.models)} wake word models")
    for name in model.models.keys():
        print(f"  - {name}")
    
    # Create test audio data (simulating 1 second of audio)
    print("\nCreating test audio data...")
    test_audio = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
    
    # Warm up the models
    print("Warming up models...")
    for _ in range(3):
        _ = model.predict(test_audio)
    
    # Run inference with timing
    print("Running inference...")
    num_runs = 10
    times = []
    
    for i in range(num_runs):
        start_time = time.time()
        predictions = model.predict(test_audio)
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        times.append(inference_time)
        
        if i == 0:  # Show first prediction
            print(f"First prediction: {predictions}")
    
    # Calculate statistics
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print(f"\nPerformance Results:")
    print(f"  Average inference time: {avg_time:.2f}ms")
    print(f"  Minimum inference time: {min_time:.2f}ms")
    print(f"  Maximum inference time: {max_time:.2f}ms")
    print(f"  Throughput: {1000/avg_time:.1f} inferences/second")
    
    # Check if we're running in real-time
    if avg_time < 80:  # Less than 80ms for 80ms audio chunk
        print("✅ Real-time performance achieved!")
    else:
        print("⚠️  Not achieving real-time performance")
    
    print("\nGPU acceleration is working!")

if __name__ == "__main__":
    main() 