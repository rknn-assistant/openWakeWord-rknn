#!/usr/bin/env python3
"""
Live Wake Word Detection for openWakeWord

A comprehensive example for real-time wake word detection using openWakeWord.
Supports multiple audio devices, device selection, and graceful error handling.

Usage:
    python3 live_wake_word_detection.py [device_number]
    python3 live_wake_word_detection.py --list

Examples:
    python3 live_wake_word_detection.py --list          # List available devices
    python3 live_wake_word_detection.py 1               # Use device 1 (USB mic)
    python3 live_wake_word_detection.py 0               # Use device 0 (built-in mic)
    python3 live_wake_word_detection.py                 # Use default device

Features:
    - Real-time audio capture and processing
    - Multiple wake word model support
    - Device selection and listing
    - Audio level monitoring
    - Graceful error handling and shutdown
    - Configurable detection parameters
"""

import time
import numpy as np
import pyaudio
from openwakeword import Model
from datetime import datetime
import sys
import signal
import argparse

# Audio settings
RATE = 16000
CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1

# Detection settings
DETECTION_INTERVAL = 0.5  # seconds
CONFIDENCE_THRESHOLD = 0.5
BUFFER_SIZE = RATE  # 1 second buffer

# Global flag for graceful shutdown
running = True

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global running
    print(f"\nShutting down gracefully...")
    running = False

def list_devices():
    """List all available audio input devices"""
    p = pyaudio.PyAudio()
    print("Available audio input devices:")
    print("=" * 50)
    
    for i in range(p.get_device_count()):
        try:
            info = p.get_device_info_by_index(i)
            max_channels = info.get('maxInputChannels', 0)
            if isinstance(max_channels, (int, float)) and max_channels > 0:
                print(f"  {i}: {info['name']}")
                print(f"      Channels: {max_channels}")
                print(f"      Sample Rate: {info.get('defaultSampleRate', 'Unknown')}")
                print()
        except Exception as e:
            print(f"  {i}: Error getting info - {e}")
    
    p.terminate()

def test_wake_word(device_index, model_name="hey_jarvis"):
    """Test wake word detection on specified device"""
    print(f"\nTesting device {device_index}")
    print("=" * 30)
    
    p = pyaudio.PyAudio()
    stream = None
    
    try:
        # Get device info
        device_info = p.get_device_info_by_index(device_index)
        print(f"Device: {device_info['name']}")
        print(f"Channels: {device_info.get('maxInputChannels', 'Unknown')}")
        print(f"Sample Rate: {device_info.get('defaultSampleRate', 'Unknown')}")
        
        # Open stream
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK,
        )
        
        print("Stream opened successfully")
        
        # Initialize model
        print("Loading wake word model...")
        model = Model()
        available_models = list(model.models.keys())
        print(f"Available models: {available_models}")
        
        if model_name not in available_models:
            print(f"Warning: Model '{model_name}' not found. Using first available model.")
            model_name = available_models[0]
        
        print(f"Testing model: {model_name}")
        
        # Audio buffer
        buffer = np.zeros(BUFFER_SIZE, dtype=np.int16)
        ptr = 0
        
        print(f"\nListening for wake word...")
        print(f"Detection threshold: {CONFIDENCE_THRESHOLD}")
        print(f"Detection interval: {DETECTION_INTERVAL}s")
        print("Press Ctrl+C to stop\n")
        
        last_detection_time = 0
        
        while running:
            try:
                # Read audio
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio = np.frombuffer(data, dtype=np.int16)
                
                # Update rolling buffer
                if ptr + CHUNK <= BUFFER_SIZE:
                    buffer[ptr:ptr+CHUNK] = audio
                    ptr += CHUNK
                else:
                    remain = BUFFER_SIZE - ptr
                    buffer[ptr:] = audio[:remain]
                    buffer[:CHUNK-remain] = audio[remain:]
                    ptr = (ptr + CHUNK) % BUFFER_SIZE
                
                # Run detection at specified interval
                current_time = time.time()
                if current_time - last_detection_time >= DETECTION_INTERVAL:
                    try:
                        preds = model.predict(buffer)
                        
                        # Handle prediction format
                        if isinstance(preds, dict):
                            conf = preds.get(model_name, 0)
                        else:
                            conf = preds[0].get(model_name, 0) if preds and len(preds) > 0 else 0
                        
                        level = np.abs(audio).mean()
                        now = datetime.now().strftime("%H:%M:%S")
                        
                        # Print status
                        status = f"[{now}] Level: {level:6.1f}  Conf: {conf:.3f}"
                        if conf > CONFIDENCE_THRESHOLD:
                            status += " 🎯 WAKE WORD DETECTED!"
                        print(status, end="\r")
                        
                        last_detection_time = current_time
                        
                    except Exception as e:
                        print(f"\nError in prediction: {e}")
                        break
                
                time.sleep(0.01)  # Small delay
                
            except KeyboardInterrupt:
                print("\n\nStopped by user")
                break
            except Exception as e:
                print(f"\nError reading audio: {e}")
                break
                
    except Exception as e:
        print(f"Error opening device {device_index}: {e}")
        print("Trying default device...")
        try:
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
            )
            print("Successfully opened default device")
            test_wake_word_default(stream, model, model_name)
        except Exception as e2:
            print(f"Failed to open default device: {e2}")
    finally:
        try:
            if stream:
                stream.stop_stream()
                stream.close()
        except:
            pass
        p.terminate()

def test_wake_word_default(stream, model, model_name):
    """Test with default device (when specific device fails)"""
    print("Testing with default device")
    print("=" * 30)
    
    buffer = np.zeros(BUFFER_SIZE, dtype=np.int16)
    ptr = 0
    last_detection_time = 0
    
    print(f"\nListening for wake word...")
    print("Press Ctrl+C to stop\n")
    
    while running:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.int16)
            
            # Update rolling buffer
            if ptr + CHUNK <= BUFFER_SIZE:
                buffer[ptr:ptr+CHUNK] = audio
                ptr += CHUNK
            else:
                remain = BUFFER_SIZE - ptr
                buffer[ptr:] = audio[:remain]
                buffer[:CHUNK-remain] = audio[remain:]
                ptr = (ptr + CHUNK) % BUFFER_SIZE
            
            # Run detection at specified interval
            current_time = time.time()
            if current_time - last_detection_time >= DETECTION_INTERVAL:
                try:
                    preds = model.predict(buffer)
                    
                    if isinstance(preds, dict):
                        conf = preds.get(model_name, 0)
                    else:
                        conf = preds[0].get(model_name, 0) if preds and len(preds) > 0 else 0
                    
                    level = np.abs(audio).mean()
                    now = datetime.now().strftime("%H:%M:%S")
                    
                    status = f"[{now}] Level: {level:6.1f}  Conf: {conf:.3f}"
                    if conf > CONFIDENCE_THRESHOLD:
                        status += " 🎯 WAKE WORD DETECTED!"
                    print(status, end="\r")
                    
                    last_detection_time = current_time
                    
                except Exception as e:
                    print(f"\nError in prediction: {e}")
                    break
            
            time.sleep(0.01)
            
        except KeyboardInterrupt:
            print("\n\nStopped by user")
            break
        except Exception as e:
            print(f"\nError reading audio: {e}")
            break

def main():
    """Main function with argument parsing"""
    global CONFIDENCE_THRESHOLD, DETECTION_INTERVAL
    
    parser = argparse.ArgumentParser(
        description="Live wake word detection with openWakeWord",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list                    # List available devices
  %(prog)s 1                         # Use device 1 (USB mic)
  %(prog)s 0                         # Use device 0 (built-in mic)
  %(prog)s                           # Use default device
        """
    )
    
    parser.add_argument(
        "device", 
        nargs="?", 
        type=int, 
        default=1,
        help="Audio device index (default: 1)"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available audio devices"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="hey_jarvis",
        help="Wake word model to use (default: hey_jarvis)"
    )
    
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=f"Detection confidence threshold (default: {CONFIDENCE_THRESHOLD})"
    )
    
    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=DETECTION_INTERVAL,
        help=f"Detection interval in seconds (default: {DETECTION_INTERVAL})"
    )
    
    args = parser.parse_args()
    
    # Update global settings
    CONFIDENCE_THRESHOLD = args.threshold
    DETECTION_INTERVAL = args.interval
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    if args.list:
        list_devices()
        return
    
    print("Live Wake Word Detection")
    print("========================")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Interval: {DETECTION_INTERVAL}s")
    print("Use --list to see available devices")
    print()
    
    test_wake_word(args.device, args.model)

if __name__ == "__main__":
    main() 