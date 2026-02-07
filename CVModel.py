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
        self._last_flat_batch: np.ndarray | None = None

    def _to_numpy(self, arr):
        if hasattr(cp, "asnumpy"):
            return cp.asnumpy(arr)
        return np.asarray(arr)


    def _run_cnn(self, inputs):
        self.cnn.addInputs(inputs)
        self.cnn.forwardPass()
        cnn_out = self.cnn.getOutputs()
        if cnn_out.ndim != 4:
            raise ValueError(f"Expected convolution output to be 4D, got shape {cnn_out.shape}")
        self._last_cnn_output_shape = cnn_out.shape
        self._last_flat_batch = self._to_numpy(cnn_out.reshape(cnn_out.shape[0], -1))
        return cnn_out

    def forwardPass(self, inputs: list[list[list[float]]]):
        cnn_out = self._run_cnn(inputs)
        batch_size = cnn_out.shape[0]
        if batch_size != 1:
            raise ValueError("forwardPass currently supports a single sample; use train_on_batch for training")

        flat = self._last_flat_batch
        if flat is None:
            raise RuntimeError("CNN outputs missing for forward pass")

        self.mlp.addInputs(flat[0].tolist())
        self.mlp.forwardPass()
        return self.mlp.neurons[-1][:]

    def train_on_batch(self, batch_inputs, expected_outputs, learningRate: float):
        if not batch_inputs or not expected_outputs:
            return None
        if len(batch_inputs) != len(expected_outputs):
            raise ValueError("Batch inputs and expected outputs must be the same length")

        batch_arr = cp.asarray(np.stack(batch_inputs), dtype=cp.float32)
        cnn_out = self._run_cnn(batch_arr)
        if self._last_flat_batch is None:
            return None

        batch_size = len(batch_inputs)
        spatial_shape = self._last_cnn_output_shape[1:] if self._last_cnn_output_shape else None
        if spatial_shape is None:
            raise RuntimeError("CNN output shape unavailable for batch training")

        grads = []
        losses = []
        scaled_lr = learningRate / max(1, batch_size)

        for sample_flat, expected in zip(self._last_flat_batch, expected_outputs):
            self.mlp.addInputs(sample_flat.tolist())
            self.mlp.forwardPass()
            mlp_grad = self.mlp.backpropagate(expected, scaled_lr)
            grads.append(np.asarray(mlp_grad, dtype=np.float32).reshape(spatial_shape))
            if self.mlp.last_loss is not None:
                losses.append(self.mlp.last_loss)

        cnn_grad = cp.asarray(np.stack(grads, axis=0), dtype=cp.float32)
        self.cnn.backpropagateFromGradient(scaled_lr, cnn_grad)

        if not losses:
            return None
        return float(sum(losses) / len(losses))
    
    def backpropigate(self, expectedOutput: list[float], learningRate: float = 1e-3):
        if self._last_cnn_output_shape is None:
            raise RuntimeError("forwardPass must be called before backpropigate")
        if self._last_cnn_output_shape[0] != 1:
            raise RuntimeError("Use train_on_batch when working with multiple samples")

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