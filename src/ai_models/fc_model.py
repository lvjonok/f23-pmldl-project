import torch

from typing import Union, Optional
from torch import nn
from src.ai_models.base import BaseAiCtrl


class FullyConnectedCtrl(BaseAiCtrl):
    def __init__(self,
                 nv: int,
                 ctrl: int,
                 linear_layers: int,
                 linear_hidden_size: int,
                 linear_dropout: float = 0,
                 linear_activation: Optional[nn.Module] = None,):
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
                    if i == 0 else
                    nn.Linear(linear_hidden_size, linear_hidden_size),
                    self.linear_activation
                )
                for i in range(linear_layers - 1)
            ],
            # Last layer
            nn.Linear(linear_hidden_size, ctrl)
            if linear_layers > 1 else
            nn.Linear(3 * nv, ctrl)
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
            raise ValueError('Input tensor should have shape (3 * nv) or (N, 3 * nv)')

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
