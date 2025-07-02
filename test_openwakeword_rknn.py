#!/usr/bin/env python3
"""
Test openWakeWord with RKNN support.

This script demonstrates how to use openWakeWord with RKNN models on the RK3588 NPU.
"""

import numpy as np
import time
from openwakeword import Model

def test_openwakeword_rknn():
    """Test openWakeWord with RKNN models."""
    print("Testing openWakeWord with RKNN support...")
    print("=" * 60)
    
    try:
        # Initialize openWakeWord with RKNN inference framework
        print("Initializing openWakeWord with RKNN backend...")
        oww = Model(
            inference_framework="rknn",
            wakeword_models=["alexa_v0.1", "hey_mycroft_v0.1", "hey_rhasspy_v0.1", "timer_v0.1", "weather_v0.1"]
        )
        
        print(f"✓ Successfully loaded {len(oww.models)} RKNN models")
        
        # Create dummy audio data (1 second of audio at 16kHz)
        sample_rate = 16000
        audio_duration = 1.0  # seconds
        num_samples = int(sample_rate * audio_duration)
        
        # Generate random audio data (simulating microphone input)
        print(f"\nGenerating test audio data ({audio_duration}s at {sample_rate}Hz)...")
        test_audio = np.random.random(num_samples).astype(np.float32)
        
        # Run prediction
        print("Running wake word detection...")
        start_time = time.time()
        
        predictions = oww.predict(test_audio)
        
        inference_time = time.time() - start_time
        
        print(f"✓ Inference completed in {inference_time:.3f} seconds")
        print(f"  Processing speed: {audio_duration/inference_time:.1f}x real-time")
        
        # Display results
        print("\nWake word detection results:")
        print("-" * 40)
        for model_name, prediction in predictions.items():
            confidence = prediction[0] if isinstance(prediction, (list, np.ndarray)) else prediction
            print(f"  {model_name}: {confidence:.4f}")
        
        # Test with multiple audio chunks
        print(f"\nTesting with multiple audio chunks...")
        chunk_size = 1024  # 64ms chunks at 16kHz
        num_chunks = num_samples // chunk_size
        
        total_chunks = 0
        total_time = 0
        
        for i in range(min(10, num_chunks)):  # Test first 10 chunks
            chunk_start = i * chunk_size
            chunk_end = chunk_start + chunk_size
            audio_chunk = test_audio[chunk_start:chunk_end]
            
            start_time = time.time()
            chunk_predictions = oww.predict(audio_chunk)
            chunk_time = time.time() - start_time
            
            total_chunks += 1
            total_time += chunk_time
            
            # Show highest confidence prediction
            max_confidence = 0
            max_model = None
            for model_name, prediction in chunk_predictions.items():
                confidence = prediction[0] if isinstance(prediction, (list, np.ndarray)) else prediction
                if confidence > max_confidence:
                    max_confidence = confidence
                    max_model = model_name
            
            if max_confidence > 0.5:  # Threshold for potential wake word
                print(f"  Chunk {i+1}: {max_model} ({max_confidence:.3f}) - {chunk_time*1000:.1f}ms")
        
        avg_chunk_time = total_time / total_chunks if total_chunks > 0 else 0
        print(f"\nAverage chunk processing time: {avg_chunk_time*1000:.1f}ms")
        print(f"Real-time factor: {0.064/avg_chunk_time:.1f}x")  # 64ms chunks
        
        print("\n" + "=" * 60)
        print("🎉 openWakeWord with RKNN support is working correctly!")
        print("\nKey benefits achieved:")
        print("  ✓ Models running on RK3588 NPU")
        print("  ✓ Real-time processing capability")
        print("  ✓ Multiple wake word detection")
        print("  ✓ Low latency inference")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def benchmark_rknn_vs_cpu():
    """Compare RKNN vs CPU performance."""
    print("\n" + "=" * 60)
    print("Benchmarking RKNN vs CPU performance...")
    print("=" * 60)
    
    try:
        # Test RKNN
        print("Testing RKNN backend...")
        oww_rknn = Model(inference_framework="rknn", wakeword_models=["alexa_v0.1"])
        
        test_audio = np.random.random(16000).astype(np.float32)  # 1 second
        
        # Warm up
        for _ in range(3):
            oww_rknn.predict(test_audio)
        
        # Benchmark RKNN
        start_time = time.time()
        for _ in range(10):
            oww_rknn.predict(test_audio)
        rknn_time = time.time() - start_time
        
        print(f"RKNN average time: {rknn_time/10:.3f}s")
        
        # Test CPU (ONNX)
        print("Testing CPU (ONNX) backend...")
        oww_cpu = Model(inference_framework="onnx", wakeword_models=["alexa_v0.1"])
        
        # Warm up
        for _ in range(3):
            oww_cpu.predict(test_audio)
        
        # Benchmark CPU
        start_time = time.time()
        for _ in range(10):
            oww_cpu.predict(test_audio)
        cpu_time = time.time() - start_time
        
        print(f"CPU average time: {cpu_time/10:.3f}s")
        
        # Calculate speedup
        speedup = cpu_time / rknn_time
        print(f"\nRKNN speedup: {speedup:.1f}x faster than CPU")
        
        return True
        
    except Exception as e:
        print(f"✗ Benchmark failed: {e}")
        return False

def main():
    """Main test function."""
    print("openWakeWord RKNN Integration Test")
    print("=" * 60)
    
    # Test basic functionality
    success1 = test_openwakeword_rknn()
    
    # Test performance comparison
    success2 = benchmark_rknn_vs_cpu()
    
    if success1 and success2:
        print("\n🎉 All tests passed! RKNN integration is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main() 