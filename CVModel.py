from Network import NeuralNetwork
from ConvolutionalNeuralNetwork_numpy import CNNVectorized
import numpy as np

try:
    import cupy as cp  # keep CNN gradients on the same backend
except ImportError:  # pragma: no cover - CPU fallback
    cp = np

class CVModel:
    def __init__(self, inputSize: tuple[int], outputs: int, layers: int, kernelsPerLayer: int, inputChannels: int, kernelSize: int):
        self.inputSize = inputSize
        self.outputs = outputs
        self.layers = layers
        self.kernelsPerLayer = kernelsPerLayer
        self.inputChannels = inputChannels
        self.kernelSize = kernelSize
        self.poolerSize = 1
        self.cnn = CNNVectorized(inputSize, inputSize, kernelSize, self.poolerSize, layers, kernelsPerLayer, inputChannels)
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

        cnn_grad = cp.asarray(mlp_grad, dtype=cp.float32).reshape(self._last_cnn_output_shape)
        self.cnn.backpropagateFromGradient(learningRate, cnn_grad)
        return self.mlp.last_loss

    def __getstate__(self):
        return {
            "inputSize": self.inputSize,
            "outputs": self.outputs,
            "layers": self.layers,
            "kernelsPerLayer": self.kernelsPerLayer,
            "inputChannels": self.inputChannels,
            "kernelSize": self.kernelSize,
            "poolerSize": self.poolerSize,
            "mlp": self.mlp,
            "_last_cnn_output_shape": self._last_cnn_output_shape,
            "cnn_state": self.cnn.__getstate__(),
        }

    def __setstate__(self, state):
        self.inputSize = tuple(state["inputSize"])
        self.outputs = state["outputs"]
        self.layers = state["layers"]
        self.kernelsPerLayer = state["kernelsPerLayer"]
        self.inputChannels = state["inputChannels"]
        self.kernelSize = state["kernelSize"]
        self.poolerSize = state.get("poolerSize", 1)
        self.mlp = state["mlp"]
        self._last_cnn_output_shape = state.get("_last_cnn_output_shape")

        self.cnn = CNNVectorized(
            self.inputSize,
            self.inputSize,
            self.kernelSize,
            self.poolerSize,
            self.layers,
            self.kernelsPerLayer,
            self.inputChannels
        )
        self.cnn.__setstate__(state["cnn_state"])