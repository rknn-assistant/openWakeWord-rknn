# Copyright 2024 openWakeWord contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for RKNN integration with openWakeWord.

These tests verify that RKNN models can be loaded and used for inference
on Rockchip NPUs (e.g., RK3588).
"""

import os
import sys
import logging
import numpy as np
import pytest
import tempfile
import time
from pathlib import Path

# Import openWakeWord
import openwakeword

# Skip all tests if RKNN is not available
try:
    from rknnlite.api import RKNNLite
    RKNN_AVAILABLE = True
except ImportError:
    RKNN_AVAILABLE = False
    RKNNLite = None

# Download models needed for tests
openwakeword.utils.download_models()


@pytest.mark.skipif(not RKNN_AVAILABLE, reason="RKNN toolkit not available")
class TestRKNNIntegration:
    """Test RKNN integration with openWakeWord."""
    
    def test_rknn_model_loading(self):
        """Test that RKNN models can be loaded successfully."""
        models_dir = Path("openwakeword/resources/models")
        
        # Test wake word models
        wakeword_models = {
            "alexa_v0.1.rknn": [1, 16, 96],
            "hey_mycroft_v0.1.rknn": [1, 16, 96],
            "hey_rhasspy_v0.1.rknn": [1, 16, 96],
            "timer_v0.1.rknn": [1, 34, 96],
            "weather_v0.1.rknn": [1, 22, 96]
        }
        
        for model_file, input_shape in wakeword_models.items():
            model_path = models_dir / model_file
            if model_path.exists():
                # Load RKNN model
                rknn_lite = RKNNLite()
                ret = rknn_lite.load_rknn(str(model_path))
                assert ret == 0, f"Failed to load RKNN model: {model_file}"
                
                # Initialize runtime environment
                ret = rknn_lite.init_runtime()
                assert ret == 0, f"Failed to init runtime: {model_file}"
                
                # Create dummy input data
                dummy_input = np.random.random(input_shape).astype(np.float32)
                
                # Run inference
                outputs = rknn_lite.inference(inputs=[dummy_input])
                
                # Verify output shape
                assert len(outputs) > 0, f"No outputs from model: {model_file}"
                assert outputs[0].shape[0] == 1, f"Unexpected output shape: {model_file}"
                
                rknn_lite.release()
    
    def test_openwakeword_rknn_framework(self):
        """Test openWakeWord with RKNN inference framework."""
        # Initialize openWakeWord with RKNN inference framework
        oww = openwakeword.Model(
            inference_framework="rknn",
            wakeword_models=["alexa_v0.1", "hey_mycroft_v0.1", "hey_rhasspy_v0.1", "timer_v0.1", "weather_v0.1"]
        )
        
        # Verify models are loaded
        assert len(oww.models) > 0, "No models loaded"
        
        # Create test audio data (1 second of audio at 16kHz)
        sample_rate = 16000
        audio_duration = 1.0  # seconds
        num_samples = int(sample_rate * audio_duration)
        test_audio = np.random.random(num_samples).astype(np.float32)
        
        # Run prediction
        predictions = oww.predict(test_audio)
        
        # Verify predictions
        assert len(predictions) > 0, "No predictions returned"
        for model_name, prediction in predictions.items():
            assert isinstance(prediction, (float, np.ndarray, list)), f"Invalid prediction type for {model_name}"
    
    def test_rknn_inference_performance(self):
        """Test RKNN inference performance."""
        # Initialize model
        oww = openwakeword.Model(
            inference_framework="rknn",
            wakeword_models=["alexa_v0.1"]
        )
        
        # Create test audio data
        test_audio = np.random.random(16000).astype(np.float32)  # 1 second
        
        # Warm up
        for _ in range(3):
            oww.predict(test_audio)
        
        # Benchmark
        num_runs = 10
        start_time = time.time()
        for _ in range(num_runs):
            oww.predict(test_audio)
        total_time = time.time() - start_time
        
        avg_time = total_time / num_runs
        print(f"Average inference time: {avg_time:.3f}s")
        
        # Verify reasonable performance (should be faster than real-time)
        assert avg_time < 1.0, f"Inference too slow: {avg_time:.3f}s"
    
    def test_rknn_vs_onnx_consistency(self):
        """Test that RKNN and ONNX models produce consistent results."""
        # Initialize both models
        oww_rknn = openwakeword.Model(
            inference_framework="rknn",
            wakeword_models=["alexa_v0.1"]
        )
        
        oww_onnx = openwakeword.Model(
            inference_framework="onnx",
            wakeword_models=["alexa_v0.1"]
        )
        
        # Create test audio data
        test_audio = np.random.random(16000).astype(np.float32)
        
        # Get predictions
        pred_rknn = oww_rknn.predict(test_audio)
        pred_onnx = oww_onnx.predict(test_audio)
        
        # Compare predictions (allow for some numerical differences)
        rknn_val = pred_rknn.get("alexa_v0.1", 0)
        onnx_val = pred_onnx.get("alexa_v0.1", 0)
        
        if isinstance(rknn_val, (list, np.ndarray)):
            rknn_val = rknn_val[0]
        if isinstance(onnx_val, (list, np.ndarray)):
            onnx_val = onnx_val[0]
        
        # Allow for reasonable tolerance due to quantization differences
        tolerance = 0.1
        assert abs(rknn_val - onnx_val) < tolerance, f"Predictions differ too much: RKNN={rknn_val:.4f}, ONNX={onnx_val:.4f}"
    
    def test_rknn_model_reset(self):
        """Test that RKNN models can be reset properly."""
        oww = openwakeword.Model(
            inference_framework="rknn",
            wakeword_models=["alexa_v0.1"]
        )
        
        # Run prediction
        test_audio = np.random.random(16000).astype(np.float32)
        predictions1 = oww.predict(test_audio)
        
        # Reset model
        oww.reset()
        
        # Run prediction again
        predictions2 = oww.predict(test_audio)
        
        # Verify reset worked (predictions should be independent)
        assert "alexa_v0.1" in predictions1
        assert "alexa_v0.1" in predictions2
    
    def test_rknn_with_multiple_models(self):
        """Test RKNN with multiple wake word models."""
        oww = openwakeword.Model(
            inference_framework="rknn",
            wakeword_models=["alexa_v0.1", "hey_mycroft_v0.1", "hey_rhasspy_v0.1"]
        )
        
        # Verify all models are loaded
        expected_models = ["alexa_v0.1", "hey_mycroft_v0.1", "hey_rhasspy_v0.1"]
        for model_name in expected_models:
            assert model_name in oww.models, f"Model {model_name} not loaded"
        
        # Test prediction
        test_audio = np.random.random(16000).astype(np.float32)
        predictions = oww.predict(test_audio)
        
        # Verify all models produced predictions
        for model_name in expected_models:
            assert model_name in predictions, f"No prediction for {model_name}"


@pytest.mark.skipif(not RKNN_AVAILABLE, reason="RKNN toolkit not available")
class TestRKNNUtilities:
    """Test RKNN utility functions."""
    
    def test_rknn_utils_import(self):
        """Test that RKNN utilities can be imported."""
        from openwakeword import rknn_utils
        assert rknn_utils is not None
    
    def test_convert_onnx_to_rknn_function_exists(self):
        """Test that RKNN conversion function exists."""
        from openwakeword.rknn_utils import convert_onnx_to_rknn
        assert callable(convert_onnx_to_rknn)
    
    def test_convert_tflite_to_rknn_function_exists(self):
        """Test that TFLite to RKNN conversion function exists."""
        from openwakeword.rknn_utils import convert_tflite_to_rknn
        assert callable(convert_tflite_to_rknn)
    
    def test_convert_openwakeword_models_to_rknn_function_exists(self):
        """Test that openWakeWord to RKNN conversion function exists."""
        from openwakeword.rknn_utils import convert_openwakeword_models_to_rknn
        assert callable(convert_openwakeword_models_to_rknn)


@pytest.mark.skipif(RKNN_AVAILABLE, reason="RKNN toolkit available - testing fallback")
class TestRKNNFallback:
    """Test fallback behavior when RKNN is not available."""
    
    def test_rknn_framework_fallback(self):
        """Test that openWakeWord falls back gracefully when RKNN is not available."""
        with pytest.raises(ValueError, match="RKNN inference framework not available"):
            openwakeword.Model(inference_framework="rknn")
    
    def test_rknn_utils_import_fallback(self):
        """Test that RKNN utilities handle missing RKNN toolkit gracefully."""
        from openwakeword import rknn_utils
        
        # Test conversion function returns False when RKNN not available
        result = rknn_utils.convert_onnx_to_rknn("dummy.onnx", "dummy.rknn")
        assert result is False 