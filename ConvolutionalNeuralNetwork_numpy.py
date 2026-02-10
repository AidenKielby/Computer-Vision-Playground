try:
    import cupy as cp
    _cupy_available = True
except ImportError:  # pragma: no cover - CPU fallback
    import numpy as cp  # type: ignore
    _cupy_available = False
import numpy as np


def _to_numpy(arr):
    if _cupy_available:
        return cp.asnumpy(arr)
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def _to_backend(arr):
    return cp.asarray(arr, dtype=cp.float32)

class CNNVectorized:

    def __init__(self, inputSize, outputSize, kernelSize, poolerSize, layers, kernelsPerLayer, inputChannels=1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.kernelSize = kernelSize
        self.poolerSize = poolerSize
        self.numLayers = layers
        self.kernelsPerLayer = kernelsPerLayer
        self.inputChannels = inputChannels

        self.inputs = None
        self.layers = []
        self.preActivation = []
        self.pooled_output = None
        self.pool_cache = None
        self.pool_stride = self._determine_pool_stride()
        if self.pool_stride is not None:
            self.poolerSize = self.pool_stride
        else:
            self.poolerSize = 1

        self.kernelLayers = []
        in_ch = self.inputChannels
        for _ in range(self.numLayers):
            self.kernelLayers.append(self._init_kernels(in_ch))
            in_ch = self.kernelsPerLayer
        self.biases = [cp.zeros((self.kernelsPerLayer,), dtype=cp.float32) for _ in range(self.numLayers)]

        self.distanceFromKernelCenter = kernelSize // 2
        self.grad_clip = 5.0

    def _determine_pool_stride(self):
        in_h, in_w = self.inputSize
        out_h, out_w = self.outputSize
        if out_h is None or out_w is None:
            return None
        if out_h == in_h and out_w == in_w:
            return None
        if out_h <= 0 or out_w <= 0:
            raise ValueError("outputSize dimensions must be positive")
        if in_h % out_h != 0 or in_w % out_w != 0:
            raise ValueError("outputSize must evenly divide inputSize for pooling")
        return (in_h // out_h, in_w // out_w)

    def _avg_pool(self, tensor):
        if self.pool_stride is None:
            self.pool_cache = None
            return tensor
        sh, sw = self.pool_stride
        B, C, H, W = tensor.shape
        out_h = H // sh
        out_w = W // sw
        reshaped = tensor.reshape(B, C, out_h, sh, out_w, sw)
        pooled = reshaped.mean(axis=(3, 5))
        self.pool_cache = (tensor.shape, sh, sw)
        return pooled

    def _avg_pool_backward(self, grad):
        if self.pool_stride is None or self.pool_cache is None:
            return grad
        orig_shape, sh, sw = self.pool_cache
        B, C, H, W = orig_shape
        out_h = H // sh
        out_w = W // sw
        grad = grad.reshape(B, C, out_h, out_w, 1, 1)
        grad = cp.broadcast_to(grad, (B, C, out_h, out_w, sh, sw))
        grad = grad / (sh * sw)
        return grad.reshape(orig_shape)

    def _init_kernels(self, in_channels):
        # per-kernel, per-channel weights
        return cp.random.uniform(-0.1, 0.1, size=(self.kernelsPerLayer, in_channels, self.kernelSize, self.kernelSize)).astype(cp.float32)

    @staticmethod
    def _leaky_relu(x):
        return cp.where(x > 0, x, 0.1 * x)

    @staticmethod
    def _leaky_relu_deriv(x):
        return cp.where(x > 0, 1.0, 0.1)

    def addInputs(self, inputs):
        arr = cp.asarray(inputs, dtype=cp.float32)
        if arr.ndim == 2:
            arr = arr[None, None, ...]
        elif arr.ndim == 3:
            arr = arr[None, ...]
        self.inputs = arr

    def _conv_same(self, inp, kernel):
        pad = self.distanceFromKernelCenter
        padded = cp.pad(inp, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
        view = cp.lib.stride_tricks.sliding_window_view(
            padded, (self.kernelSize, self.kernelSize), axis=(2, 3)
        )
        return cp.tensordot(view, kernel, axes=((1, -1, -2), (0, 1, 2)))  # (B, H, W)

    def _kernel_grad(self, prev_maps, grad_map, kernel):
        pad = self.distanceFromKernelCenter
        padded = cp.pad(prev_maps, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
        view = cp.lib.stride_tricks.sliding_window_view(
            padded, (self.kernelSize, self.kernelSize), axis=(2, 3)
        )
        k_grad = cp.sum(view * grad_map[:, None, ..., None, None], axis=(0, 2, 3))  # (C, ks, ks)
        # average over batch
        k_grad /= prev_maps.shape[0]
        return k_grad

    def _prev_grad(self, grad_maps, kernels, prev_shape):
        pad = self.distanceFromKernelCenter
        B = prev_shape[0]
        C_prev = prev_shape[1]
        H_prev, W_prev = prev_shape[2], prev_shape[3]
        prev_grad = cp.zeros((B, C_prev, H_prev, W_prev), dtype=cp.float32)
        padded = cp.pad(grad_maps, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")  # (B, K, H+2p, W+2p)
        view = cp.lib.stride_tricks.sliding_window_view(
            padded, (self.kernelSize, self.kernelSize), axis=(2, 3)
        )
        flipped = cp.flip(kernels, axis=(2, 3))  # (K, C_prev, ks, ks)
        for c in range(C_prev):
            prev_grad[:, c] += cp.sum(view * flipped[None, :, None, None, c, :, :], axis=(1, 4, 5))
        return prev_grad

    def forwardPass(self):
        assert self.inputs is not None, "Call addInputs first"
        self.layers = []
        self.preActivation = []
        self.pooled_output = None
        self.pool_cache = None

        prev = self.inputs
        for layer_idx in range(self.numLayers):
            kernels = self.kernelLayers[layer_idx]
            biases = self.biases[layer_idx]
            out_list = []
            pre_list = []
            for k_idx in range(self.kernelsPerLayer):
                kernel = kernels[k_idx]
                pre = self._conv_same(prev, kernel) + biases[k_idx]
                act = self._leaky_relu(pre)
                pre_list.append(pre)
                out_list.append(act)
            prev = cp.stack(out_list, axis=1)
            self.preActivation.append(cp.stack(pre_list, axis=1))
            self.layers.append(prev)

        self.pooled_output = self._avg_pool(prev)

    def getOutputs(self):
        if self.pooled_output is not None:
            return self.pooled_output
        if self.layers:
            return self.layers[-1]
        raise RuntimeError("No CNN outputs available; call forwardPass first")

    def backpropagate(self, correctOutput, learningRate):
        expected = cp.asarray(correctOutput, dtype=cp.float32)
        if expected.ndim == 2:
            expected = expected[None, None, ...]
        elif expected.ndim == 3:
            expected = expected[None, ...]
        out = self.getOutputs()
        if expected.shape != out.shape:
            raise ValueError("expected output must have same shape as network output")
        pooled_grad = cp.clip(out - expected, -10.0, 10.0)
        front_grad = self._avg_pool_backward(pooled_grad)
        deriv = self._leaky_relu_deriv(self.preActivation[-1])
        front_grad = cp.clip(front_grad * deriv, -10.0, 10.0)
        for layer_idx in reversed(range(self.numLayers)):
            prev_acts = self.inputs if layer_idx == 0 else self.layers[layer_idx-1]
            kernels = self.kernelLayers[layer_idx]
            kernel_grads = cp.zeros_like(kernels)
            bias_grads = cp.zeros_like(self.biases[layer_idx])

            for k_idx in range(self.kernelsPerLayer):
                kernel_grads[k_idx] = self._kernel_grad(prev_acts, front_grad[:, k_idx], kernels[k_idx])
                bias_grads[k_idx] = cp.sum(front_grad[:, k_idx]) / front_grad.shape[0]

            k_norm = cp.linalg.norm(kernel_grads)
            if k_norm > self.grad_clip:
                kernel_grads *= self.grad_clip / (k_norm + 1e-8)
            self.kernelLayers[layer_idx] -= learningRate * kernel_grads
            self.biases[layer_idx] -= learningRate * bias_grads

            if layer_idx > 0:
                prev_grad = self._prev_grad(front_grad, kernels, prev_acts.shape)
                prev_deriv = self._leaky_relu_deriv(self.preActivation[layer_idx-1])
                front_grad = cp.clip(prev_grad * prev_deriv, -10.0, 10.0)


    def backpropagateFromGradient(self, learningRate, gradient):
        """
        outputGradient shape:
            (batch_size, kernelsPerLayer, height, width)

        Represents:
            dLoss / d(CNN_output_activation)
        """
        if self.pool_stride is not None:
            gradient = self._avg_pool_backward(gradient)
        frontGradient = gradient * self._leaky_relu_deriv(self.preActivation[-1])

        for layerIndex in reversed(range(self.numLayers)):
            previousActivations = (
                self.inputs if layerIndex == 0 else self.layers[layerIndex - 1]
            )

            currentKernels = self.kernelLayers[layerIndex]

            kernelGradients = cp.zeros_like(currentKernels)
            biasGradients = cp.zeros_like(self.biases[layerIndex])

            for kernelIndex in range(self.kernelsPerLayer):
                kernelGradients[kernelIndex] = self._kernel_grad(
                    previousActivations,
                    frontGradient[:, kernelIndex],
                    currentKernels[kernelIndex]
                )

                biasGradients[kernelIndex] = cp.mean(frontGradient[:, kernelIndex])

            if layerIndex > 0:
                previousGradient = self._prev_grad(
                    frontGradient,
                    currentKernels,
                    previousActivations.shape
                )
                frontGradient = (
                    previousGradient *
                    self._leaky_relu_deriv(self.preActivation[layerIndex - 1])
                )

            kernelNorm = cp.linalg.norm(kernelGradients)
            if kernelNorm > self.grad_clip:
                kernelGradients *= self.grad_clip / (kernelNorm + 1e-8)

            self.kernelLayers[layerIndex] -= learningRate * kernelGradients
            self.biases[layerIndex] -= learningRate * biasGradients

    def __getstate__(self):
        return {
            "inputSize": self.inputSize,
            "outputSize": self.outputSize,
            "kernelSize": self.kernelSize,
            "poolerSize": self.poolerSize,
            "numLayers": self.numLayers,
            "kernelsPerLayer": self.kernelsPerLayer,
            "inputChannels": self.inputChannels,
            "kernelLayers": [_to_numpy(k) for k in self.kernelLayers],
            "biases": [_to_numpy(b) for b in self.biases],
            "grad_clip": self.grad_clip,
        }

    def __setstate__(self, state):
        self.inputSize = tuple(state["inputSize"])
        self.outputSize = tuple(state["outputSize"])
        self.kernelSize = state["kernelSize"]
        self.poolerSize = state["poolerSize"]
        self.numLayers = state["numLayers"]
        self.kernelsPerLayer = state["kernelsPerLayer"]
        self.inputChannels = state["inputChannels"]
        self.grad_clip = state.get("grad_clip", 5.0)

        self.inputs = None
        self.layers = []
        self.preActivation = []

        self.kernelLayers = [_to_backend(k) for k in state["kernelLayers"]]
        self.biases = [_to_backend(b) for b in state["biases"]]
        self.distanceFromKernelCenter = self.kernelSize // 2
        self.pool_stride = self._determine_pool_stride()
        if self.pool_stride is not None:
            self.poolerSize = self.pool_stride
        else:
            self.poolerSize = 1
        self.pool_cache = None
        self.pooled_output = None

def train_on_symbols_vec(iterations=1000, lr=0.01):
    x_img = cp.array([
        [1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1]
    ], dtype=cp.float32)
    check_img = cp.array([
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=cp.float32)

    net = CNNVectorized(inputSize=(5, 5), outputSize=(5, 5), kernelSize=3, poolerSize=1, layers=5, kernelsPerLayer=10)

    for it in range(iterations):
        for img in (x_img, check_img):
            net.addInputs(img)
            net.forwardPass()
            net.backpropagate(img, learningRate=lr)
        if (it + 1) % max(1, iterations // 10) == 0:
            print(f"iter {it+1}/{iterations}")

    for name, img in (("X", x_img), ("Check", check_img)):
        net.addInputs(img)
        net.forwardPass()
        out = net.getOutputs()[0]
        print(f"{name} output:")
        for row in out:
            print([f"{v:.2f}" for v in row])
        print()

if __name__ == "__main__":
    cp.random.seed(0)
    train_on_symbols_vec(iterations=500, lr=0.05)
