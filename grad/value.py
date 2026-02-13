import math
from typing import Callable


class Value:
    data: float
    grad: float
    _backward: Callable[[], None]
    _prev: set["Value"]
    _op: str
    label: str

    def __init__(
        self,
        data: float,
        _children: tuple["Value", ...] = (),
        _op: str = "",
        label: str = "",
    ) -> None:
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other: "int | float | Value") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __mul__(self, other: "int | float | Value") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other: int | float) -> "Value":
        assert isinstance(other, (int, float)), "only supports int/float"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad

        out._backward = _backward

        return out

    def __radd__(self, other: int | float) -> "Value":  # other + self
        return self + other

    def __rmul__(self, other: int | float) -> "Value":  # other * self
        return self * other

    def __truediv__(self, other: int | float) -> "Value":  # self / other
        return self * other**-1

    def __rtruediv__(self, other: int | float) -> "Value":  # other / self
        return other * self**-1

    def __neg__(self) -> "Value":
        return self * -1

    def __sub__(self, other) -> "Value":
        return self + (-other)

    def tanh(self) -> "Value":
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward

        return out

    def exp(self) -> "Value":
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            self.grad = out.data * out.grad

        out._backward = _backward

        return out

    def backward(self):
        topo: list["Value"] = []
        visited: set["Value"] = set()

        # build list from children to root
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        # go from root to children b/c
        # children need parent for chain rule
        for node in reversed(topo):
            node._backward()
