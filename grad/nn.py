import random
from value import Value


class Neuron:
    activation: str
    w: list[Value]
    b: Value

    def __init__(self, nin: int, activation: str) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]  # weight
        self.b = Value(random.uniform(-1, 1))  # bias
        self.activation = activation

    def __call__(self, x):  # forard pass
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)  # dot prod
        return getattr(act, self.activation)()  # activation function

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer:
    neurons: list[Neuron]

    def __init__(self, nin: int, nout: int, activation: str) -> None:
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]  # pass input to all neurons in layer
        return outs  # outs[0] if len(outs) == 1 else outs  # prettier print

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(
        self, nin: int, nouts: list[int], activations: list[str] | None = None
    ):  # num of inputs to sys, nums of outs per layer
        activations = (
            ["tanh" for _ in range(len(nouts))] if activations is None else activations
        )
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i + 1], activations[i]) for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
