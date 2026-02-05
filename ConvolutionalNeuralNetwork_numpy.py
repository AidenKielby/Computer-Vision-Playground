import cupy as cp

class CNNVectorized:

    def __init__(self, inputSize, outputSize, kernelSize, poolerSize, layers, kernelsPerLayer, inputChannels=1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.kernelSize = kernelSize
        self.poolerSize = poolerSize
        # match original: numLayers = layers + 1
        self.numLayers = layers + 1
        self.kernelsPerLayer = kernelsPerLayer
        self.inputChannels = inputChannels

        self.inputs = None
        self.layers = []
        self.preActivation = []

        self.kernelLayers = []
        in_ch = self.inputChannels
        for _ in range(self.numLayers):
            self.kernelLayers.append(self._init_kernels(in_ch))
            in_ch = self.kernelsPerLayer
        self.biases = [cp.zeros((self.kernelsPerLayer,), dtype=cp.float32) for _ in range(self.numLayers)]

        self.distanceFromKernelCenter = kernelSize // 2
        self.grad_clip = 5.0

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

    def getOutputs(self):
        return self.layers[-1]

    def backpropagate(self, correctOutput, learningRate):
        expected = cp.asarray(correctOutput, dtype=cp.float32)
        if expected.ndim == 2:
            expected = expected[None, None, ...]
        elif expected.ndim == 3:
            expected = expected[None, ...]
        out = self.layers[-1]
        deriv = self._leaky_relu_deriv(self.preActivation[-1])
        if expected.shape[0] != out.shape[0]:
            raise ValueError("expected output must have same channel count as network output")
        front_grad = cp.clip((out - expected) * deriv, -10.0, 10.0)
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
