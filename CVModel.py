from Network import NeuralNetwork
from ConvolutionalNeuralNetwork_numpy import CNNVectorized
import numpy as np

class CVModel:
    def __init__(self, inputSize: tuple[int], outputs: int, layers: int, kernelsPerLayer: int, inputChannels: int, kernelSize: int):
        self.inputSize = inputSize
        self.outputs = outputs
        self.layers = layers
        self.kernelsPerLayer = kernelsPerLayer
        self.inputChannels = inputChannels
        self.kernelSize = kernelSize
        self.cnn = CNNVectorized(inputSize, inputSize, kernelSize, 1, layers, kernelsPerLayer, inputChannels)
        self.mlp = NeuralNetwork(kernelsPerLayer * inputSize[0] * inputSize[1], 1, 32, outputs)
        self._last_cnn_output_shape: tuple[int, ...] | None = None


    def forwardPass(self, inputs: list[list[list[float]]]):
        self.cnn.addInputs(inputs)
        self.cnn.forwardPass()
        cnn_out = self.cnn.getOutputs()
        if cnn_out.ndim != 4:
            raise ValueError(f"Expected convolution output to be 4D, got shape {cnn_out.shape}")

        self._last_cnn_output_shape = cnn_out.shape
        batch_size = cnn_out.shape[0]
        flat = cnn_out.reshape(batch_size, -1)

        if batch_size != 1:
            raise ValueError("Current MLP implementation only supports a batch size of 1")

        self.mlp.addInputs(flat[0].tolist())
        self.mlp.forwardPass()
        return self.mlp.neurons[-1][:]
    
    def backpropigate(self, expectedOutput: list[float], learningRate: float = 1e-3):
        if self._last_cnn_output_shape is None:
            raise RuntimeError("forwardPass must be called before backpropigate")

        mlp_grad = self.mlp.backpropagate(expectedOutput, learningRate)

        expected_size = np.prod(self._last_cnn_output_shape)
        if len(mlp_grad) != expected_size:
            raise ValueError(
                f"Gradient size {len(mlp_grad)} does not match convolution output size {expected_size}"
            )

        cnn_grad = np.asarray(mlp_grad, dtype=np.float32).reshape(self._last_cnn_output_shape)
        self.cnn.backpropagateFromGradient(learningRate, cnn_grad)
        return self.mlp.last_loss