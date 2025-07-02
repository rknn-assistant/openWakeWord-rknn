#!/usr/bin/env python3
"""
Test script for RKNN wake word models.

This script tests the converted RKNN models to ensure they work correctly.
"""

import numpy as np
import os
from rknnlite.api import RKNNLite

def test_rknn_model(model_path, input_shape, model_name):
    """Test a single RKNN model."""
    try:
        print(f"\nTesting {model_name}...")
        
        # Load RKNN model
        rknn_lite = RKNNLite()
        ret = rknn_lite.load_rknn(model_path)
        if ret != 0:
            print(f"Failed to load RKNN model: {ret}")
            return False
        
        # Initialize runtime environment
        ret = rknn_lite.init_runtime()
        if ret != 0:
            print(f"Failed to init runtime: {ret}")
            return False
        
        # Create dummy input data
        dummy_input = np.random.random(input_shape).astype(np.float32)
        
        # Run inference
        outputs = rknn_lite.inference(inputs=[dummy_input])
        
        print(f"✓ {model_name} inference successful!")
        print(f"  Input shape: {input_shape}")
        print(f"  Output shape: {outputs[0].shape}")
        print(f"  Output values: {outputs[0].flatten()[:5]}...")  # Show first 5 values
        
        return True
        
    except Exception as e:
        print(f"✗ {model_name} test failed: {e}")
        return False
    finally:
        if 'rknn_lite' in locals():
            rknn_lite.release()

def main():
    """Test all converted RKNN models."""
    models_dir = "openwakeword/resources/models"
    
    # Test wake word models
    wakeword_models = {
        "alexa_v0.1.rknn": [1, 16, 96],
        "hey_mycroft_v0.1.rknn": [1, 16, 96],
        "hey_rhasspy_v0.1.rknn": [1, 16, 96],
        "timer_v0.1.rknn": [1, 34, 96],
        "weather_v0.1.rknn": [1, 22, 96]
    }
    
    print("Testing RKNN wake word models...")
    print("=" * 50)
    
    success_count = 0
    total_count = len(wakeword_models)
    
    for model_file, input_shape in wakeword_models.items():
        model_path = os.path.join(models_dir, model_file)
        if os.path.exists(model_path):
            if test_rknn_model(model_path, input_shape, model_file):
                success_count += 1
        else:
            print(f"✗ {model_file} not found")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {success_count}/{total_count} models working")
    
    if success_count == total_count:
        print("🎉 All RKNN models are working correctly!")
    else:
        print("⚠️  Some models failed. Check the errors above.")

if __name__ == "__main__":
    main() 