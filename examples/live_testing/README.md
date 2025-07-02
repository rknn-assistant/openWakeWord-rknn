# Live Testing Examples

This directory contains examples for real-time wake word detection using openWakeWord.

## Overview

The live testing examples demonstrate how to use openWakeWord for real-time audio processing and wake word detection. These examples are designed to work with microphones and provide real-time feedback on audio levels and detection confidence.

## Files

### `live_wake_word_detection.py`
The main live wake word detection script with comprehensive features:

- **Real-time audio capture** from microphone devices
- **Multiple device support** with device selection and listing
- **Configurable detection parameters** (threshold, interval)
- **Multiple wake word model support**
- **Graceful error handling** and shutdown
- **Audio level monitoring**

## Usage

### Basic Usage

```bash
# List available audio devices
python3 live_wake_word_detection.py --list

# Use default device (device 1)
python3 live_wake_word_detection.py

# Use specific device
python3 live_wake_word_detection.py 1  # USB microphone
python3 live_wake_word_detection.py 0  # Built-in microphone
```

### Advanced Usage

```bash
# Use different wake word model
python3 live_wake_word_detection.py --model alexa

# Adjust detection threshold
python3 live_wake_word_detection.py --threshold 0.7

# Change detection interval
python3 live_wake_word_detection.py --interval 0.25

# Combine options
python3 live_wake_word_detection.py 1 --model hey_jarvis --threshold 0.6 --interval 0.3
```

## Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--list` | `-l` | List available audio devices | - |
| `--model` | `-m` | Wake word model to use | `hey_jarvis` |
| `--threshold` | `-t` | Detection confidence threshold | `0.5` |
| `--interval` | `-i` | Detection interval in seconds | `0.5` |

## Available Models

The following wake word models are available by default:

- `hey_jarvis` - "Hey Jarvis" wake word
- `alexa` - "Alexa" wake word  
- `hey_mycroft` - "Hey Mycroft" wake word
- `hey_rhasspy` - "Hey Rhasspy" wake word
- `timer` - Timer wake word
- `weather` - Weather wake word

## Audio Device Selection

### Listing Devices
Use `--list` to see all available audio input devices:

```bash
python3 live_wake_word_detection.py --list
```

Example output:
```
Available audio input devices:
==================================================
  0: realtek,rt5616-codec: fe470000.i2s-rt5616-aif1 rt5616-aif1-0 (hw:0,0)
      Channels: 2
      Sample Rate: 44100.0

  1: USB Camera-B4.09.24.1: Audio (hw:1,0)
      Channels: 4
      Sample Rate: 16000.0
```

### Device Recommendations

- **Device 1 (USB mic)**: Usually provides better audio quality and correct sample rate (16kHz)
- **Device 0 (Built-in mic)**: May have different sample rates, check device info
- **Default device**: System default, fallback when specific device fails

## Output Format

The script provides real-time output showing:

```
[21:15:22] Level:  293.4  Conf: 0.000
[21:15:23] Level:  156.7  Conf: 0.823 🎯 WAKE WORD DETECTED!
```

Where:
- `Level`: Audio input level (higher = louder)
- `Conf`: Detection confidence (0.0 to 1.0)
- `🎯 WAKE WORD DETECTED!`: Appears when confidence exceeds threshold

## Troubleshooting

### Common Issues

1. **"Error opening device"**: Try using `--list` to see available devices
2. **Low audio levels**: Check microphone permissions and volume settings
3. **No detections**: Try lowering the threshold with `--threshold 0.3`
4. **High CPU usage**: Increase detection interval with `--interval 1.0`

### Audio Setup

Ensure your system has:
- Working microphone hardware
- Proper audio drivers installed
- Microphone permissions granted to Python
- Appropriate volume levels

### Performance Tips

- Use USB microphones for better audio quality
- Adjust detection interval based on your needs (lower = more responsive, higher = less CPU)
- Monitor audio levels to ensure adequate input signal
- Use appropriate confidence thresholds for your environment

## Examples

### Quick Start
```bash
# Start with USB microphone
python3 live_wake_word_detection.py 1
```

### High Sensitivity
```bash
# Lower threshold for more sensitive detection
python3 live_wake_word_detection.py --threshold 0.3 --interval 0.25
```

### Multiple Models
```bash
# Test different wake words
python3 live_wake_word_detection.py --model alexa
python3 live_wake_word_detection.py --model hey_mycroft
```

## Dependencies

Required packages:
- `openwakeword` - Main wake word detection library
- `pyaudio` - Audio capture
- `numpy` - Numerical processing

Install with:
```bash
pip install openwakeword pyaudio numpy
```

## License

This code is part of the openWakeWord project and follows the same license terms. 