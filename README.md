# Computer-Vision-Playground

This is probably one of the projects I have cared about the most. It fuses my convolutional network from the (currently shelved) diffusion-model experiments with the multilayer perceptron from my earlier neural-network work, creating a playground for building simple computer-vision ideas.

The gui is purposely bad, however currently not the 'bad' im looking for. My plan is to make it look more like the clasic Tkinter GUI, because for some reason i like it.

### Prerequisites
- Python 3.9 or higher
- Webcam
- (Optional) CUDA-capable GPU for acceleration

## What it does
- Captures webcam frames and trains a model on the fly with user-labeled targets.
- Chains a NumPy/CuPy-style CNN front-end to a hand-built MLP classifier.
- Lets me prototype different spatial resolutions, kernel counts, and output spaces without regenerating scaffolding.

## Current status
- **Live training works**: the Train screen streams the camera feed, you pick the “correct output,” hit start, and gradients flow every frame.
- **Model creation works**: tweak kernel sizes, depth, and class counts, then jump straight into training mode.
- **Loading/saving is in progress**: the UI hooks exist, but serialization still needs to be finished.

## Quick start
1. `python ComputerVisionPlayground.py`
2. Choose **Create New**, dial in the conv/MLP settings, and submit.
3. click **Submit**, aim the webcam, select the class label, and toggle live training.

### GPU Acceleration (Optional)

To use CuPy for GPU acceleration:

```bash
# For CUDA 12.x
pip install cupy-cuda12x

# Then edit ConvolutionalNeuralNetwork_numpy.py:
# Change: import numpy as cp
# To:     import cupy as cp
```

## Dependencies

- **PySide6**: GUI framework
- **OpenCV**: Webcam capture and image processing
- **NumPy**: Numerical computing for neural networks
- **CuPy** (optional): GPU-accelerated computing

## Next up
- Change the trining to sample videos based on the users input, and train X epochs on those videos. Move live loop trining to loading.
- Finish the save/load path so experiments aren’t strictly in-memory.
- Add better telemetry (loss plots, per-class confidence readouts) to understand what the live loop is learning.

- Explore lightweight data augmentation for more stable real-time training.


