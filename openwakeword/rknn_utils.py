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
RKNN utilities for openWakeWord model conversion and optimization.

This module provides utilities to convert openWakeWord models to RKNN format
for deployment on Rockchip NPUs (e.g., RK3588).
"""

import os
import logging
import numpy as np
from typing import List, Optional, Dict, Any
import tempfile
import shutil

try:
    from rknn.api import RKNN
except ImportError:
    RKNN = None
    logging.warning("RKNN toolkit not found. Install it to use RKNN conversion features.")


def convert_onnx_to_rknn(
    onnx_model_path: str,
    output_path: str,
    target_platform: str = "rk3588",
    quantization: bool = True,
    optimization_level: int = 3,
    **kwargs
) -> bool:
    """
    Convert an ONNX model to RKNN format for deployment on Rockchip NPUs.
    
    Args:
        onnx_model_path: Path to the input ONNX model
        output_path: Path where the RKNN model will be saved
        target_platform: Target platform (e.g., "rk3588", "rk3568")
        quantization: Whether to enable INT8 quantization
        optimization_level: RKNN optimization level (0-3)
        **kwargs: Additional arguments passed to RKNN build
        
    Returns:
        bool: True if conversion successful, False otherwise
    """
    if RKNN is None:
        logging.error("RKNN toolkit not available. Please install it first.")
        return False
    
    if not os.path.exists(onnx_model_path):
        logging.error(f"ONNX model not found: {onnx_model_path}")
        return False
    
    try:
        # Create RKNN object
        rknn = RKNN(verbose=True)
        
        # Pre-process config
        logging.info(f"Configuring RKNN for {target_platform}")
        ret = rknn.config(target_platform=target_platform, dynamic_input=True)
        if ret != 0:
            logging.error(f"RKNN config failed: {ret}")
            return False
            
        logging.info(f"Loading ONNX model: {onnx_model_path}")
        ret = rknn.load_onnx(model=onnx_model_path)
        if ret != 0:
            logging.error(f"Load ONNX model failed: {ret}")
            return False
        
        # RKNN model build
        logging.info(f"Building RKNN model for {target_platform}")
        ret = rknn.build(
            do_quantization=quantization,
            dataset=None,  # No calibration dataset for now
            rknn_batch_size=1,
            target_platform=target_platform,
            optimization_level=optimization_level,
            **kwargs
        )
        if ret != 0:
            logging.error(f"Build RKNN model failed: {ret}")
            return False
        
        # Export RKNN model
        logging.info(f"Exporting RKNN model to: {output_path}")
        ret = rknn.export_rknn(output_path)
        if ret != 0:
            logging.error(f"Export RKNN model failed: {ret}")
            return False
        
        logging.info("RKNN model conversion completed successfully!")
        return True
        
    except Exception as e:
        logging.error(f"RKNN conversion failed: {e}")
        return False
    finally:
        if 'rknn' in locals():
            rknn.release()


def convert_tflite_to_rknn(
    tflite_model_path: str,
    output_path: str,
    target_platform: str = "rk3588",
    quantization: bool = True,
    optimization_level: int = 3,
    **kwargs
) -> bool:
    """
    Convert a TensorFlow Lite model to RKNN format for deployment on Rockchip NPUs.
    
    Args:
        tflite_model_path: Path to the input TFLite model
        output_path: Path where the RKNN model will be saved
        target_platform: Target platform (e.g., "rk3588", "rk3568")
        quantization: Whether to enable INT8 quantization
        optimization_level: RKNN optimization level (0-3)
        **kwargs: Additional arguments passed to RKNN build
        
    Returns:
        bool: True if conversion successful, False otherwise
    """
    if RKNN is None:
        logging.error("RKNN toolkit not available. Please install it first.")
        return False
    
    if not os.path.exists(tflite_model_path):
        logging.error(f"TFLite model not found: {tflite_model_path}")
        return False
    
    try:
        # Create RKNN object
        rknn = RKNN(verbose=True)
        
        # Pre-process config
        logging.info(f"Configuring RKNN for {target_platform}")
        ret = rknn.config(target_platform=target_platform, dynamic_input=True)
        if ret != 0:
            logging.error(f"RKNN config failed: {ret}")
            return False
            
        logging.info(f"Loading TFLite model: {tflite_model_path}")
        ret = rknn.load_tflite(model=tflite_model_path)
        if ret != 0:
            logging.error(f"Load TFLite model failed: {ret}")
            return False
        
        # RKNN model build
        logging.info(f"Building RKNN model for {target_platform}")
        ret = rknn.build(
            do_quantization=quantization,
            dataset=None,  # No calibration dataset for now
            rknn_batch_size=1,
            target_platform=target_platform,
            optimization_level=optimization_level,
            **kwargs
        )
        if ret != 0:
            logging.error(f"Build RKNN model failed: {ret}")
            return False
        
        # Export RKNN model
        logging.info(f"Exporting RKNN model to: {output_path}")
        ret = rknn.export_rknn(output_path)
        if ret != 0:
            logging.error(f"Export RKNN model failed: {ret}")
            return False
        
        logging.info("RKNN model conversion completed successfully!")
        return True
        
    except Exception as e:
        logging.error(f"RKNN conversion failed: {e}")
        return False
    finally:
        if 'rknn' in locals():
            rknn.release()


def convert_openwakeword_models_to_rknn(
    model_names: Optional[List[str]] = None,
    target_platform: str = "rk3588",
    quantization: bool = True,
    output_directory: Optional[str] = None,
    **kwargs
) -> Dict[str, bool]:
    """
    Convert openWakeWord models to RKNN format.
    
    Args:
        model_names: List of model names to convert. If None, converts all available models.
        target_platform: Target platform (e.g., "rk3588", "rk3568")
        quantization: Whether to enable INT8 quantization
        output_directory: Directory to save RKNN models. If None, uses default location.
        **kwargs: Additional arguments passed to RKNN build
        
    Returns:
        Dict[str, bool]: Dictionary mapping model names to conversion success status
    """
    import openwakeword
    
    if output_directory is None:
        output_directory = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "resources", "models"
        )
    
    os.makedirs(output_directory, exist_ok=True)
    
    # Get model paths
    if model_names is None:
        model_names = list(openwakeword.MODELS.keys())
    
    results = {}
    
    for model_name in model_names:
        if model_name not in openwakeword.MODELS:
            logging.warning(f"Model {model_name} not found in available models")
            results[model_name] = False
            continue
        
        # Get ONNX model path
        onnx_path = openwakeword.MODELS[model_name]["model_path"].replace(".tflite", ".onnx")
        if not os.path.exists(onnx_path):
            logging.warning(f"ONNX model not found for {model_name}: {onnx_path}")
            results[model_name] = False
            continue
        
        # Convert to RKNN
        rknn_path = os.path.join(output_directory, f"{model_name}.rknn")
        success = convert_onnx_to_rknn(
            onnx_path, 
            rknn_path, 
            target_platform=target_platform,
            quantization=quantization,
            **kwargs
        )
        results[model_name] = success
    
    return results


def convert_feature_models_to_rknn(
    target_platform: str = "rk3588",
    quantization: bool = True,
    output_directory: Optional[str] = None,
    **kwargs
) -> Dict[str, bool]:
    """
    Convert openWakeWord feature models (melspectrogram and embedding) to RKNN format.
    
    Args:
        target_platform: Target platform (e.g., "rk3588", "rk3568")
        quantization: Whether to enable INT8 quantization
        output_directory: Directory to save RKNN models. If None, uses default location.
        **kwargs: Additional arguments passed to RKNN build
        
    Returns:
        Dict[str, bool]: Dictionary mapping model names to conversion success status
    """
    import openwakeword
    
    if output_directory is None:
        output_directory = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "resources", "models"
        )
    
    os.makedirs(output_directory, exist_ok=True)
    
    results = {}
    
    # Convert melspectrogram model
    melspec_onnx = openwakeword.FEATURE_MODELS["melspectrogram"]["model_path"].replace(".tflite", ".onnx")
    if os.path.exists(melspec_onnx):
        melspec_rknn = os.path.join(output_directory, "melspectrogram.rknn")
        results["melspectrogram"] = convert_onnx_to_rknn(
            melspec_onnx, 
            melspec_rknn, 
            target_platform=target_platform,
            quantization=quantization,
            **kwargs
        )
    else:
        logging.warning(f"Melspectrogram ONNX model not found: {melspec_onnx}")
        results["melspectrogram"] = False
    
    # Convert embedding model
    embedding_onnx = openwakeword.FEATURE_MODELS["embedding"]["model_path"].replace(".tflite", ".onnx")
    if os.path.exists(embedding_onnx):
        embedding_rknn = os.path.join(output_directory, "embedding_model.rknn")
        results["embedding"] = convert_onnx_to_rknn(
            embedding_onnx, 
            embedding_rknn, 
            target_platform=target_platform,
            quantization=quantization,
            **kwargs
        )
    else:
        logging.warning(f"Embedding ONNX model not found: {embedding_onnx}")
        results["embedding"] = False
    
    return results


def benchmark_rknn_model(
    rknn_model_path: str,
    input_data: np.ndarray,
    num_runs: int = 100
) -> Dict[str, float]:
    """
    Benchmark RKNN model performance.
    
    Args:
        rknn_model_path: Path to the RKNN model
        input_data: Input data for benchmarking
        num_runs: Number of inference runs for averaging
        
    Returns:
        Dict[str, float]: Dictionary containing benchmark results
    """
    try:
        from rknnlite.api import RKNNLite
    except ImportError:
        logging.error("RKNNLite not available for benchmarking")
        return {}
    
    try:
        # Load and initialize RKNN model
        rknn = RKNNLite()
        ret = rknn.load_rknn(rknn_model_path)
        if ret != 0:
            logging.error(f"Failed to load RKNN model: {ret}")
            return {}
        
        ret = rknn.init_runtime()
        if ret != 0:
            logging.error(f"Failed to init RKNN runtime: {ret}")
            return {}
        
        # Warm up
        for _ in range(10):
            rknn.inference(inputs=[input_data])
        
        # Benchmark
        import time
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            rknn.inference(inputs=[input_data])
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        return {
            "avg_inference_time_ms": avg_time,
            "std_inference_time_ms": std_time,
            "min_inference_time_ms": min_time,
            "max_inference_time_ms": max_time,
            "throughput_fps": 1000.0 / avg_time if avg_time > 0 else 0
        }
        
    except Exception as e:
        logging.error(f"Benchmarking failed: {e}")
        return {}
    finally:
        if 'rknn' in locals():
            rknn.release()


def validate_rknn_model(
    original_model_path: str,
    rknn_model_path: str,
    test_input: np.ndarray,
    tolerance: float = 1e-3
) -> bool:
    """
    Validate RKNN model against original model.
    
    Args:
        original_model_path: Path to original ONNX/TFLite model
        rknn_model_path: Path to RKNN model
        test_input: Test input data
        tolerance: Tolerance for output comparison
        
    Returns:
        bool: True if models produce similar outputs, False otherwise
    """
    try:
        # Get original model output
        original_output = None
        if original_model_path.endswith('.onnx'):
            import onnxruntime as ort
            session = ort.InferenceSession(original_model_path)
            original_output = session.run(None, {session.get_inputs()[0].name: test_input})[0]
        elif original_model_path.endswith('.tflite'):
            import tflite_runtime.interpreter as tflite
            interpreter = tflite.Interpreter(model_path=original_model_path)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
            original_output = interpreter.get_tensor(output_details[0]['index'])
        
        # Get RKNN model output
        from rknnlite.api import RKNNLite
        rknn = RKNNLite()
        ret = rknn.load_rknn(rknn_model_path)
        if ret != 0:
            logging.error(f"Failed to load RKNN model: {ret}")
            return False
        
        ret = rknn.init_runtime()
        if ret != 0:
            logging.error(f"Failed to init RKNN runtime: {ret}")
            return False
        
        rknn_output = rknn.inference(inputs=[test_input])[0]
        
        # Compare outputs
        if original_output is None or rknn_output is None:
            return False
        
        diff = np.abs(original_output - rknn_output)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        logging.info(f"Model validation - Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
        
        return max_diff < tolerance
        
    except Exception as e:
        logging.error(f"Model validation failed: {e}")
        return False
    finally:
        if 'rknn' in locals():
            rknn.release() 