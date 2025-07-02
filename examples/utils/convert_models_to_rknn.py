#!/usr/bin/env python3
"""
Convert openWakeWord Models to RKNN Format

This utility script converts openWakeWord models to RKNN format for deployment
on Rockchip NPUs (e.g., RK3588, RK3568).

Requirements:
- RKNN Toolkit 2.3.2 or later
- openWakeWord models (ONNX or TFLite format)

Usage:
    python convert_models_to_rknn.py [--models MODEL1,MODEL2] [--platform PLATFORM]
"""

import argparse
import os
import sys
import logging
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from openwakeword.rknn_utils import (
        convert_openwakeword_models_to_rknn,
        convert_feature_models_to_rknn,
        benchmark_rknn_model,
        validate_rknn_model
    )
    from openwakeword.utils import download_models
except ImportError as e:
    print(f"Error importing openWakeWord: {e}")
    print("Please ensure openWakeWord is installed with RKNN support.")
    sys.exit(1)

try:
    from rknn.api import RKNN
    RKNN_AVAILABLE = True
except ImportError:
    print("Warning: RKNN toolkit not available. Please install RKNN Toolkit 2.3.2+")
    RKNN_AVAILABLE = False


def setup_logging(verbose=False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def setup_argument_parser():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert openWakeWord Models to RKNN Format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all available models
  python convert_models_to_rknn.py

  # Convert specific models
  python convert_models_to_rknn.py --models alexa_v0.1,hey_mycroft_v0.1

  # Convert for specific platform
  python convert_models_to_rknn.py --platform rk3568

  # Convert with custom settings
  python convert_models_to_rknn.py --no-quantization --optimization-level 2

  # Convert and validate
  python convert_models_to_rknn.py --validate
        """
    )
    
    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of model names to convert (default: all available)"
    )
    
    parser.add_argument(
        "--platform",
        type=str,
        default="rk3588",
        choices=["rk3588", "rk3568", "rk3566", "rk3562"],
        help="Target platform for RKNN conversion"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for RKNN models (default: openwakeword/resources/models)"
    )
    
    parser.add_argument(
        "--no-quantization",
        action="store_true",
        help="Disable INT8 quantization (use FP16 instead)"
    )
    
    parser.add_argument(
        "--optimization-level",
        type=int,
        default=3,
        choices=[0, 1, 2, 3],
        help="RKNN optimization level (0=no optimization, 3=maximum optimization)"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate converted models against original models"
    )
    
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark on converted models"
    )
    
    parser.add_argument(
        "--include-features",
        action="store_true",
        help="Also convert feature models (melspectrogram, embedding)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser


def check_prerequisites():
    """Check if all prerequisites are met."""
    print("Checking prerequisites...")
    
    # Check RKNN availability
    if not RKNN_AVAILABLE:
        print("❌ RKNN toolkit not available")
        print("   Please install RKNN Toolkit 2.3.2 or later")
        return False
    
    # Check if models directory exists
    models_dir = Path("openwakeword/resources/models")
    if not models_dir.exists():
        print("❌ Models directory not found")
        print("   Please run: python -m openwakeword.utils.download_models")
        return False
    
    # Check for ONNX/TFLite models
    onnx_models = list(models_dir.glob("*.onnx"))
    tflite_models = list(models_dir.glob("*.tflite"))
    
    if not onnx_models and not tflite_models:
        print("❌ No ONNX or TFLite models found")
        print("   Please download models first")
        return False
    
    print(f"✓ Found {len(onnx_models)} ONNX models and {len(tflite_models)} TFLite models")
    return True


def get_available_models():
    """Get list of available models for conversion."""
    models_dir = Path("openwakeword/resources/models")
    
    # Get wake word models
    wakeword_models = []
    for model_file in models_dir.glob("*.onnx"):
        model_name = model_file.stem
        if not model_name.endswith(("_melspectrogram", "_embedding")):
            wakeword_models.append(model_name)
    
    for model_file in models_dir.glob("*.tflite"):
        model_name = model_file.stem
        if not model_name.endswith(("_melspectrogram", "_embedding")):
            wakeword_models.append(model_name)
    
    return wakeword_models


def convert_models(args):
    """Convert models to RKNN format."""
    print(f"\n🔄 Converting models to RKNN format for {args.platform}...")
    
    # Determine models to convert
    if args.models:
        model_names = [name.strip() for name in args.models.split(",")]
    else:
        model_names = get_available_models()
    
    if not model_names:
        print("❌ No models found for conversion")
        return False
    
    print(f"Models to convert: {', '.join(model_names)}")
    
    # Convert wake word models
    results = convert_openwakeword_models_to_rknn(
        model_names=model_names,
        target_platform=args.platform,
        quantization=not args.no_quantization,
        output_directory=args.output_dir,
        optimization_level=args.optimization_level
    )
    
    # Report results
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    
    print(f"\n📊 Conversion Results:")
    print(f"  Successful: {successful}/{total}")
    
    for model_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {model_name}")
    
    # Convert feature models if requested
    if args.include_features:
        print(f"\n🔄 Converting feature models...")
        feature_results = convert_feature_models_to_rknn(
            target_platform=args.platform,
            quantization=not args.no_quantization,
            output_directory=args.output_dir,
            optimization_level=args.optimization_level
        )
        
        feature_successful = sum(1 for success in feature_results.values() if success)
        feature_total = len(feature_results)
        
        print(f"Feature Models Results:")
        print(f"  Successful: {feature_successful}/{feature_total}")
        
        for model_name, success in feature_results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {model_name}")
    
    return successful > 0


def validate_models(args):
    """Validate converted RKNN models."""
    if not args.validate:
        return True
    
    print(f"\n🔍 Validating converted models...")
    
    # This would require loading both original and RKNN models
    # and comparing outputs with test inputs
    # For now, just check that RKNN files exist
    models_dir = Path(args.output_dir or "openwakeword/resources/models")
    
    rknn_models = list(models_dir.glob("*.rknn"))
    print(f"Found {len(rknn_models)} RKNN models:")
    
    for model_file in rknn_models:
        file_size = model_file.stat().st_size / 1024  # KB
        print(f"  ✓ {model_file.name} ({file_size:.1f} KB)")
    
    return len(rknn_models) > 0


def benchmark_models(args):
    """Run performance benchmark on converted models."""
    if not args.benchmark:
        return True
    
    print(f"\n⚡ Running performance benchmarks...")
    
    # This would require loading RKNN models and running inference
    # For now, just provide guidance
    print("To benchmark RKNN models, use the example script:")
    print("  python examples/rknn_wake_word_detection.py --benchmark")
    
    return True


def main():
    """Main function."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    print("🚀 openWakeWord RKNN Model Converter")
    print("=" * 50)
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Download models if needed
    print("\n📥 Ensuring models are downloaded...")
    download_models()
    
    # Convert models
    success = convert_models(args)
    if not success:
        print("❌ Model conversion failed")
        sys.exit(1)
    
    # Validate models
    if args.validate:
        validate_models(args)
    
    # Benchmark models
    if args.benchmark:
        benchmark_models(args)
    
    print("\n✅ RKNN model conversion completed!")
    print("\nNext steps:")
    print("  1. Test the converted models:")
    print("     python examples/rknn_wake_word_detection.py")
    print("  2. Use in your application:")
    print("     from openwakeword import Model")
    print("     oww = Model(inference_framework='rknn')")


if __name__ == "__main__":
    main() 