#!/usr/bin/env python3
"""
Convert openWakeWord models to RKNN format for RK3588 NPU deployment.

This script converts the ONNX models to RKNN format with proper input shape specifications.
"""

import os
import sys
import logging
from rknn.api import RKNN

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_model_to_rknn(onnx_path, rknn_path, input_shapes, target_platform="rk3588"):
    """
    Convert a single ONNX model to RKNN format.
    
    Args:
        onnx_path: Path to ONNX model
        rknn_path: Path to save RKNN model
        input_shapes: List of input shapes for the model
        target_platform: Target platform
    """
    try:
        # Create RKNN object
        rknn = RKNN(verbose=True)
        
        # Configure RKNN
        logger.info(f"Configuring RKNN for {target_platform}")
        ret = rknn.config(target_platform=target_platform)
        if ret != 0:
            logger.error(f"RKNN config failed: {ret}")
            return False
        
        # Load ONNX model with input shapes
        logger.info(f"Loading ONNX model: {onnx_path}")
        ret = rknn.load_onnx(model=onnx_path, input_size_list=input_shapes)
        if ret != 0:
            logger.error(f"Load ONNX model failed: {ret}")
            return False
        
        # Build RKNN model
        logger.info(f"Building RKNN model")
        ret = rknn.build(do_quantization=False)  # No quantization for now
        if ret != 0:
            logger.error(f"Build RKNN model failed: {ret}")
            return False
        
        # Export RKNN model
        logger.info(f"Exporting RKNN model to: {rknn_path}")
        ret = rknn.export_rknn(rknn_path)
        if ret != 0:
            logger.error(f"Export RKNN model failed: {ret}")
            return False
        
        logger.info(f"Successfully converted {onnx_path} to {rknn_path}")
        return True
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return False
    finally:
        if 'rknn' in locals():
            rknn.release()

def main():
    """Main conversion function."""
    # Define model paths and input shapes
    models_dir = "openwakeword/resources/models"
    
    # Feature models (melspectrogram and embedding)
    feature_models = {
        "melspectrogram.onnx": {
            "input_shapes": [[1, 1280]],  # [batch_size, samples]
            "output": "melspectrogram.rknn"
        },
        "embedding_model.onnx": {
            "input_shapes": [[1, 76, 32, 1]],  # [batch_size, frames, features, channels]
            "output": "embedding_model.rknn"
        }
    }
    
    # Wake word models
    wakeword_models = {
        "alexa_v0.1.onnx": {
            "input_shapes": [[1, 16, 96]],  # [batch_size, frames, features]
            "output": "alexa_v0.1.rknn"
        },
        "hey_mycroft_v0.1.onnx": {
            "input_shapes": [[1, 16, 96]],
            "output": "hey_mycroft_v0.1.rknn"
        },
        "hey_jarvis_v0.1.onnx": {
            "input_shapes": [[1, 16, 96]],
            "output": "hey_jarvis_v0.1.rknn"
        },
        "hey_rhasspy_v0.1.onnx": {
            "input_shapes": [[1, 16, 96]],
            "output": "hey_rhasspy_v0.1.rknn"
        },
        "timer_v0.1.onnx": {
            "input_shapes": [[1, 16, 96]],
            "output": "timer_v0.1.rknn"
        },
        "weather_v0.1.onnx": {
            "input_shapes": [[1, 16, 96]],
            "output": "weather_v0.1.rknn"
        }
    }
    
    # Convert feature models first
    logger.info("Converting feature models...")
    for model_name, config in feature_models.items():
        onnx_path = os.path.join(models_dir, model_name)
        rknn_path = os.path.join(models_dir, config["output"])
        
        if os.path.exists(onnx_path):
            success = convert_model_to_rknn(
                onnx_path, 
                rknn_path, 
                config["input_shapes"]
            )
            if success:
                logger.info(f"✓ {model_name} converted successfully")
            else:
                logger.error(f"✗ {model_name} conversion failed")
        else:
            logger.warning(f"ONNX model not found: {onnx_path}")
    
    # Convert wake word models
    logger.info("Converting wake word models...")
    for model_name, config in wakeword_models.items():
        onnx_path = os.path.join(models_dir, model_name)
        rknn_path = os.path.join(models_dir, config["output"])
        
        if os.path.exists(onnx_path):
            success = convert_model_to_rknn(
                onnx_path, 
                rknn_path, 
                config["input_shapes"]
            )
            if success:
                logger.info(f"✓ {model_name} converted successfully")
            else:
                logger.error(f"✗ {model_name} conversion failed")
        else:
            logger.warning(f"ONNX model not found: {onnx_path}")
    
    logger.info("Conversion process completed!")

if __name__ == "__main__":
    main() 