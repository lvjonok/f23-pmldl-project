import torch

from typing import Any, Union, Optional
from torch import nn
from src.ai_models.base import BaseAiCtrl
import casadi as cs
from csnn import set_sym_type, Linear, Sequential, ReLU, Module
from csnn.module import SymType


class FullyConnectedCtrl(BaseAiCtrl):
    def __init__(
        self,
        nv: int,
        ctrl: int,
        linear_layers: int,
        linear_hidden_size: int,
        linear_dropout: float = 0,
        linear_activation: Optional[nn.Module] = None,
    ):
        """
        Initialize the FullyConnectedCtrl model.

        Parameters:
            nv (int): Number of generalized velocities in the input.
            ctrl (int): Number of generalized controls.
            linear_layers (int): Number of linear layers in the fully connected part.
            linear_hidden_size (int): Number of features in the hidden layers of the fully connected part.
            linear_dropout (float, optional): Dropout rate for the linear layers.
            linear_activation (Optional[nn.Module], optional): Activation function for the linear layers.

        Raises:
            ValueError: If an unsupported RNN cell type is provided.
        """

        super(FullyConnectedCtrl, self).__init__()
        self.nv = nv
        self.dropout = nn.Dropout(p=linear_dropout)
        self.linear_activation = linear_activation or nn.Sigmoid()

        # Linear part
        self.linear = nn.Sequential(
            # Hidden linear layers
            *[
                nn.Sequential(
                    nn.Linear(3 * nv, linear_hidden_size)
                    if i == 0
                    else nn.Linear(linear_hidden_size, linear_hidden_size),
                    self.linear_activation,
                )
                for i in range(linear_layers - 1)
            ],
            # Last layer
            nn.Linear(linear_hidden_size, ctrl)
            if linear_layers > 1
            else nn.Linear(3 * nv, ctrl),
        )
        self._dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FullyConnectedCtrl model.

        Parameters:
            x (torch.Tensor): Input tensor of shape (3 * nv) or (N, 3 * nv).

        Returns:
            torch.Tensor: Output tensor.
        """
        if x.shape[-1] != 3 * self.nv:
            raise ValueError("Input tensor should have shape (3 * nv) or (N, 3 * nv)")

        return self.dropout(self.linear(x))

    def _make_predictions(self, x: torch.Tensor, **_) -> torch.Tensor:
        """
        Make predictions using the FullyConnectedCtrl model.

        Parameters:
            x (torch.Tensor): Input tensor for prediction.

        Returns:
            torch.Tensor: Model predictions.
        """

        with torch.no_grad():
            result = self(x)

        return result


set_sym_type("SX")  # can set either MX or SX


def sigmoid(input: SymType) -> SymType:
    """Applies the element-wise function `Sigmoid(x) = 1 / (1 + exp(-x))`."""
    return 1 / (1 + cs.exp(-input))


class Sigmoid(Module[SymType]):
    """Applies the element-wise function `Sigmoid(x) = 1 / (1 + exp(-x))`."""

    def forward(self, input: SymType) -> SymType:
        return sigmoid(input)


class CasadiModel:
    """
    CasadiModel is a wrapper around the FullyConnectedCtrl model that creates
    a casadi interface for inference in optimization problems.
    """

    def __init__(self, fcnn: FullyConnectedCtrl) -> None:
        self.fcnn = fcnn

        # copy the structure of fcnn into casadi model
        self.casadi_layers, self.weights = self.__parse_sequential(fcnn.linear)

        self.casadi_net = Sequential[cs.SX](self.casadi_layers)
        self.parameters = [v[1] for v in list(self.casadi_net.parameters())]

        self.input = cs.SX.sym("input", 1, 3 * self.fcnn.nv)
        self.full_network = cs.Function(
            "net", [self.input] + list(self.parameters), [self.casadi_net(self.input)]
        )

        self.inference = cs.Function(
            "inference", [self.input], [self.full_network(self.input, *self.weights)]
        )

    def __call__(self, x):
        return self.inference(x)

    def __parse_sequential(self, seq: torch.nn.Sequential):
        layers = []
        weights = []
        for layer in seq:
            if isinstance(layer, nn.Linear):
                layers.append(Linear(layer.in_features, layer.out_features))
                weights.append(layer.weight.detach().numpy())
                weights.append(layer.bias.detach().numpy())
            elif isinstance(layer, nn.Sequential):
                res = self.__parse_sequential(layer)
                layers.extend(res[0])
                weights.extend(res[1])
            elif isinstance(layer, nn.Sigmoid):
                layers.append(Sigmoid())
            else:
                raise NotImplementedError(f"Layer type {type(layer)} is not supported.")
        return layers, weights
