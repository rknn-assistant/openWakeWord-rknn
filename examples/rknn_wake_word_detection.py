#!/usr/bin/env python3
"""
RKNN Wake Word Detection Example

This example demonstrates how to use openWakeWord with RKNN models
for hardware-accelerated wake word detection on Rockchip NPUs (e.g., RK3588).

Requirements:
- RKNN Toolkit 2.3.2 or later
- Rockchip NPU (RK3588, RK3568, etc.)
- openWakeWord with RKNN support

Usage:
    python rknn_wake_word_detection.py [--models MODEL1,MODEL2] [--threshold THRESHOLD]
"""

import argparse
import numpy as np
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from openwakeword import Model
    from openwakeword.rknn_utils import benchmark_rknn_model
except ImportError as e:
    print(f"Error importing openWakeWord: {e}")
    print("Please ensure openWakeWord is installed with RKNN support.")
    sys.exit(1)

try:
    from rknnlite.api import RKNNLite
    RKNN_AVAILABLE = True
except ImportError:
    print("Warning: RKNN toolkit not available. This example requires RKNN for NPU acceleration.")
    RKNN_AVAILABLE = False


def setup_argument_parser():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="RKNN Wake Word Detection Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default models
  python rknn_wake_word_detection.py

  # Use specific models
  python rknn_wake_word_detection.py --models alexa_v0.1,hey_mycroft_v0.1

  # Set custom threshold
  python rknn_wake_word_detection.py --threshold 0.7

  # Benchmark mode
  python rknn_wake_word_detection.py --benchmark
        """
    )
    
    parser.add_argument(
        "--models",
        type=str,
        default="alexa_v0.1,hey_mycroft_v0.1,hey_rhasspy_v0.1",
        help="Comma-separated list of wake word models to use"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for wake word detection (0.0-1.0)"
    )
    
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark instead of interactive detection"
    )
    
    parser.add_argument(
        "--audio-file",
        type=str,
        help="Path to audio file for testing (WAV format)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Audio chunk size for processing (default: 1024 samples)"
    )
    
    return parser


def initialize_rknn_model(model_names):
    """Initialize openWakeWord with RKNN inference framework."""
    print("Initializing RKNN wake word detection...")
    
    # Parse model names
    models = [name.strip() for name in model_names.split(",")]
    
    try:
        # Initialize openWakeWord with RKNN backend
        oww = Model(
            inference_framework="rknn",
            wakeword_models=models
        )
        
        print(f"✓ Successfully loaded {len(oww.models)} RKNN models:")
        for model_name in oww.models.keys():
            print(f"  • {model_name}")
        
        return oww
        
    except Exception as e:
        print(f"✗ Failed to initialize RKNN model: {e}")
        return None


def benchmark_performance(oww, audio_data, num_runs=100):
    """Benchmark RKNN model performance."""
    print(f"\nRunning performance benchmark ({num_runs} runs)...")
    
    # Warm up
    for _ in range(5):
        oww.predict(audio_data[:16000])  # Use first second
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_runs):
        oww.predict(audio_data[:16000])
    total_time = time.time() - start_time
    
    avg_time = total_time / num_runs
    audio_duration = 1.0  # 1 second of audio
    real_time_factor = audio_duration / avg_time
    
    print(f"Performance Results:")
    print(f"  Average inference time: {avg_time*1000:.2f}ms")
    print(f"  Real-time factor: {real_time_factor:.1f}x")
    print(f"  Throughput: {1/avg_time:.1f} inferences/second")
    
    return avg_time, real_time_factor


def detect_wake_words(oww, audio_data, threshold, chunk_size):
    """Detect wake words in audio data."""
    print(f"\nDetecting wake words (threshold: {threshold})...")
    
    # Process audio in chunks
    num_chunks = len(audio_data) // chunk_size
    detections = []
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        chunk = audio_data[start_idx:end_idx]
        
        # Get predictions
        predictions = oww.predict(chunk)
        
        # Check for wake word detections
        for model_name, confidence in predictions.items():
            if isinstance(confidence, (list, np.ndarray)):
                confidence = confidence[0]
            
            if confidence >= threshold:
                timestamp = i * chunk_size / 16000  # Convert to seconds
                detections.append({
                    'model': model_name,
                    'confidence': confidence,
                    'timestamp': timestamp
                })
                print(f"🎯 {model_name} detected at {timestamp:.2f}s (confidence: {confidence:.3f})")
    
    return detections


def load_audio_file(file_path):
    """Load audio file for testing."""
    try:
        import wave
        import struct
        
        with wave.open(file_path, 'rb') as wav_file:
            # Read audio data
            frames = wav_file.readframes(wav_file.getnframes())
            audio_data = struct.unpack(f'{len(frames)//2}h', frames)
            audio_data = np.array(audio_data, dtype=np.float32) / 32768.0
            
            print(f"✓ Loaded audio file: {file_path}")
            print(f"  Duration: {len(audio_data)/16000:.2f} seconds")
            print(f"  Sample rate: {wav_file.getframerate()} Hz")
            
            return audio_data
            
    except Exception as e:
        print(f"✗ Failed to load audio file: {e}")
        return None


def generate_test_audio(duration=5.0, sample_rate=16000):
    """Generate test audio data."""
    num_samples = int(duration * sample_rate)
    
    # Generate random audio (simulating microphone input)
    audio_data = np.random.random(num_samples).astype(np.float32)
    
    # Add some structure to make it more realistic
    t = np.linspace(0, duration, num_samples)
    audio_data += 0.1 * np.sin(2 * np.pi * 440 * t)  # Add 440Hz tone
    
    print(f"✓ Generated {duration}s test audio at {sample_rate}Hz")
    return audio_data


def main():
    """Main function."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    print("🚀 RKNN Wake Word Detection Example")
    print("=" * 50)
    
    # Check RKNN availability
    if not RKNN_AVAILABLE:
        print("❌ RKNN toolkit not available. Please install RKNN Toolkit 2.3.2+")
        sys.exit(1)
    
    # Initialize model
    oww = initialize_rknn_model(args.models)
    if oww is None:
        sys.exit(1)
    
    # Load or generate audio data
    if args.audio_file:
        audio_data = load_audio_file(args.audio_file)
        if audio_data is None:
            sys.exit(1)
    else:
        audio_data = generate_test_audio()
    
    # Run benchmark or detection
    if args.benchmark:
        benchmark_performance(oww, audio_data)
    else:
        # Detect wake words
        detections = detect_wake_words(oww, audio_data, args.threshold, args.chunk_size)
        
        # Summary
        print(f"\nDetection Summary:")
        print(f"  Total detections: {len(detections)}")
        if detections:
            for detection in detections:
                print(f"  • {detection['model']}: {detection['confidence']:.3f} at {detection['timestamp']:.2f}s")
        else:
            print("  No wake words detected")
    
    print("\n✅ RKNN wake word detection completed!")


if __name__ == "__main__":
    main() 