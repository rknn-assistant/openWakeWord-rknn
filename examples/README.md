# Examples

Included are several example scripts demonstrating the usage of openWakeWord. Some of these examples have specific requirements, which are detailed below.

## Detect From Microphone

This is a simple example which allows you to test openWakeWord by using a locally connected microphone. To run the script, follow these steps:

1) Install the example-specific requirements: `pip install pyaudio`

2) Run the script: `python detect_from_microphone.py`.

Note that if you have more than one microphone connected to your system, you may need to adjust the PyAudio configuration in the script to select the appropriate input device.

## Capture Activations

This script is designed to run in the background and capture activations for the included pre-trained models. You can specify the initialization arguments, activation threshold, and output directory for the saved audio files for each activation. To run the script, follow these steps:

1) Install the example-specific requirements:

```
# On Linux
pip install pyaudio scipy

# On Windows
pip install PyAudioWPatch scipy
```

2) Run the script: `python capture_activations.py --threshold 0.5 --output_dir <my_dir> --model <my_model>`

Where `--output_dir` is the desired location to save the activation clips, and `--model` is the model name or full path of the model to use.
If `--model` is not provided, all of the default models will be loaded. Use `python capture_activations.py --help` for more information on all of the possible arguments.

Note that if you have more than one microphone connected to your system, you may need to adjust the PyAudio configuration in the script to select the appropriate input device.

## Benchmark Efficiency

This is a script that estimates how many openWakeWord models could be run on on the specified number of cores for the current system. Can be useful to determine if a given system has the resources required for a particular use-case.

To run the script: `python benchmark_efficiency.py --ncores <desired integer number of cores>`

## RKNN Wake Word Detection

This example demonstrates how to use openWakeWord with RKNN models for hardware-accelerated wake word detection on Rockchip NPUs (e.g., RK3588, RK3568).

### Requirements

- RKNN Toolkit 2.3.2 or later
- Rockchip NPU (RK3588, RK3568, etc.)
- openWakeWord with RKNN support

### Usage

Basic usage with default models:
```bash
python rknn_wake_word_detection.py
```

Use specific models:
```bash
python rknn_wake_word_detection.py --models alexa_v0.1,hey_mycroft_v0.1
```

Set custom threshold:
```bash
python rknn_wake_word_detection.py --threshold 0.7
```

Run performance benchmark:
```bash
python rknn_wake_word_detection.py --benchmark
```

Test with audio file:
```bash
python rknn_wake_word_detection.py --audio-file test_audio.wav
```

### Features

- Hardware acceleration on Rockchip NPUs
- Real-time wake word detection
- Performance benchmarking
- Support for multiple wake word models
- Audio file processing
- Configurable detection thresholds

## Live Testing

The `live_testing/` directory contains comprehensive examples for real-time wake word detection with microphones.

### Live Wake Word Detection

A feature-rich script for real-time wake word detection with multiple device support, configurable parameters, and graceful error handling.

#### Features

- **Real-time audio capture** from microphone devices
- **Multiple device support** with device selection and listing
- **Configurable detection parameters** (threshold, interval)
- **Multiple wake word model support**
- **Graceful error handling** and shutdown
- **Audio level monitoring**

#### Usage

```bash
# List available audio devices
python3 live_testing/live_wake_word_detection.py --list

# Use default device (device 1)
python3 live_testing/live_wake_word_detection.py

# Use specific device
python3 live_testing/live_wake_word_detection.py 1  # USB microphone
python3 live_testing/live_wake_word_detection.py 0  # Built-in microphone

# Advanced usage
python3 live_testing/live_wake_word_detection.py --model alexa --threshold 0.7 --interval 0.25
```

#### Requirements

```bash
pip install pyaudio numpy
```

For detailed documentation, see [live_testing/README.md](live_testing/README.md).

## Model Conversion Utilities

### Convert Models to RKNN Format

The `utils/convert_models_to_rknn.py` script converts openWakeWord models to RKNN format for deployment on Rockchip NPUs.

#### Requirements

- RKNN Toolkit 2.3.2 or later
- openWakeWord models (ONNX or TFLite format)

#### Usage

Convert all available models:
```bash
python utils/convert_models_to_rknn.py
```

Convert specific models:
```bash
python utils/convert_models_to_rknn.py --models alexa_v0.1,hey_mycroft_v0.1
```

Convert for specific platform:
```bash
python utils/convert_models_to_rknn.py --platform rk3568
```

Convert with custom settings:
```bash
python utils/convert_models_to_rknn.py --no-quantization --optimization-level 2
```

Convert and validate:
```bash
python utils/convert_models_to_rknn.py --validate
```

#### Features

- Batch conversion of multiple models
- Platform-specific optimization
- INT8 quantization support
- Model validation
- Performance benchmarking
- Support for feature models
