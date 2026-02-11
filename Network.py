import numpy as np


class NeuralNetwork:
    def __init__(self, inputNeurons: int, hiddenLayers: int, neuronsPerHiddenLayer: int, outputNeurons: int):
        self.inputNeurons = inputNeurons
        self.hiddenLayers = hiddenLayers
        self.neuronsPerHiddenLayer = neuronsPerHiddenLayer
        self.outputNeurons = outputNeurons
        self.dtype = np.float32
        self._rng = np.random.default_rng()

        self.neurons = self.initNeurons(inputNeurons, hiddenLayers, neuronsPerHiddenLayer, outputNeurons)
        self.weights = self.initWeights(inputNeurons, hiddenLayers, neuronsPerHiddenLayer, outputNeurons)
        self.biases = self.initBiases(hiddenLayers, neuronsPerHiddenLayer, outputNeurons)

        self.preActivations: list[list[float]] = [[] for _ in range(self.hiddenLayers + 1)]
        self._activations: list[np.ndarray] = []
        self._z_values: list[np.ndarray] = []
        self._input_vector: np.ndarray | None = None
        self.last_loss: float | None = None

    def _he_limit(self, fan_in: int) -> float:
        return np.sqrt(2.0 / max(1, fan_in))

    def _leaky_relu(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, 0.1 * x)

    def _leaky_relu_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1.0, 0.1)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        shifted = x - np.max(x)
        exps = np.exp(shifted)
        return exps / np.sum(exps)

    def addInputs(self, inputs: list[float]):
        if len(inputs) != self.inputNeurons:
            raise ValueError("Input length does not match network input size")
        self.neurons[0] = inputs[:]
        self._input_vector = np.asarray(inputs, dtype=self.dtype)

    def initNeurons(self, inputNeurons: int, hiddenLayers: int, neuronsPerHiddenLayer: int, outputNeurons: int):
        neurons = [[0.0 for _ in range(inputNeurons)]]
        for _ in range(hiddenLayers):
            neurons.append([0.0 for _ in range(neuronsPerHiddenLayer)])
        neurons.append([0.0 for _ in range(outputNeurons)])
        return neurons

    def initWeights(self, inputNeurons: int, hiddenLayers: int, neuronsPerHiddenLayer: int, outputNeurons: int):
        weights: list[np.ndarray] = []
        fan_in = inputNeurons
        for _ in range(hiddenLayers):
            limit = self._he_limit(fan_in)
            layer = self._rng.uniform(-limit, limit, size=(neuronsPerHiddenLayer, fan_in)).astype(self.dtype)
            weights.append(layer)
            fan_in = neuronsPerHiddenLayer
        limit = self._he_limit(fan_in)
        weights.append(self._rng.uniform(-limit, limit, size=(outputNeurons, fan_in)).astype(self.dtype))
        return weights

    def initBiases(self, hiddenLayers: int, neuronsPerHiddenLayer: int, outputNeurons: int):
        biases: list[np.ndarray] = []
        for _ in range(hiddenLayers):
            biases.append(np.full((self.neuronsPerHiddenLayer,), 0.01, dtype=np.float32))
        biases.append(np.full((self.outputNeurons,), 0.01, dtype=np.float32))
        return biases

    def forwardPass(self):
        if self._input_vector is None:
            raise RuntimeError("Inputs must be added before calling forwardPass")

        activations: list[np.ndarray] = [self._input_vector]
        z_values: list[np.ndarray] = []

        for layer_idx, (weights, bias) in enumerate(zip(self.weights, self.biases)):
            z = weights @ activations[-1] + bias
            z_values.append(z)
            if layer_idx == len(self.weights) - 1:
                if self.outputNeurons == 1:
                    a = self._sigmoid(z)
                else:
                    a = self._softmax(z)
            else:
                a = self._leaky_relu(z)
            activations.append(a.astype(self.dtype))

        self._activations = activations
        self._z_values = z_values

        for idx, act in enumerate(self._activations):
            self.neurons[idx] = act.tolist()
        self.preActivations = [z.tolist() for z in self._z_values]

    def MSE(self, networkOutput, actualAnswer):
        output = np.asarray(networkOutput, dtype=self.dtype)
        target = np.asarray(actualAnswer, dtype=self.dtype)
        diff = output - target
        return float(np.mean(diff * diff))

    def BCE(self, output, target):
        out = np.clip(np.asarray(output, dtype=self.dtype), 1e-9, 1 - 1e-9)
        tgt = np.asarray(target, dtype=self.dtype)
        return float(-np.sum(tgt * np.log(out) + (1 - tgt) * np.log(1 - out)))

    def _categorical_cross_entropy(self, output, target):
        out = np.clip(np.asarray(output, dtype=self.dtype), 1e-9, 1.0)
        tgt = np.asarray(target, dtype=self.dtype)
        return float(-np.sum(tgt * np.log(out)))

    def _softmax_cross_entropy_delta(self, output, target):
        """Gradient of softmax + categorical cross-entropy combo."""
        return output - target

    def backpropagate(self, correctOutput: list[float], learningRate: float):
        if not self._activations or not self._z_values:
            raise RuntimeError("forwardPass must be called before backpropagate")

        target = np.asarray(correctOutput, dtype=self.dtype)
        output = self._activations[-1]
        if target.shape != output.shape:
            target = target.reshape(output.shape)

        if self.outputNeurons == 1:
            self.last_loss = self.BCE(output, target)
            delta = output - target
        else:
            self.last_loss = self._categorical_cross_entropy(output, target)
            delta = self._softmax_cross_entropy_delta(output, target)

        input_grad = np.zeros(self.inputNeurons, dtype=self.dtype)

        for layer_idx in reversed(range(len(self.weights))):
            weights = self.weights[layer_idx]
            a_prev = self._activations[layer_idx]

            grad_w = np.outer(delta, a_prev)
            grad_b = delta

            if layer_idx == 0:
                input_grad = weights.T @ delta
            else:
                delta = (weights.T @ delta)
                if layer_idx < len(self.weights):
                    delta = delta * self._leaky_relu_derivative(self._z_values[layer_idx - 1])

            self.weights[layer_idx] = weights - learningRate * grad_w
            self.biases[layer_idx] -= learningRate * grad_b

        return input_grad.tolist()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_rng", None)
        state.pop("_activations", None)
        state.pop("_z_values", None)
        state.pop("_input_vector", None)
        return state

    def __setstate__(self, state):
        self.inputNeurons = state.get("inputNeurons")
        self.hiddenLayers = state.get("hiddenLayers", 0)
        self.neuronsPerHiddenLayer = state.get("neuronsPerHiddenLayer", 0)
        self.outputNeurons = state.get("outputNeurons", 0)
        self.dtype = np.float32
        self._rng = np.random.default_rng()

        weights_state = state.get("weights", [])
        self.weights = [np.asarray(w, dtype=self.dtype) for w in weights_state]
        biases_state = state.get("biases", [])
        self.biases = [np.asarray(b, dtype=self.dtype) for b in biases_state]

        if "neurons" in state and state["neurons"]:
            self.neurons = state["neurons"]
        else:
            self.neurons = self.initNeurons(self.inputNeurons, self.hiddenLayers, self.neuronsPerHiddenLayer, self.outputNeurons)

        self.preActivations = state.get("preActivations", [[] for _ in range(self.hiddenLayers + 1)])
        self.last_loss = state.get("last_loss")

        self._activations = []
        self._z_values = []
        self._input_vector = None

