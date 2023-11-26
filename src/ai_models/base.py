import torch

from typing import Union, Optional
from torch import nn
from torch.utils.data import DataLoader


def to_device(_device: torch.device, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tuple(t.to(_device) for t in tensors)


def train_one_epoch(
        train_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        train_loaded: DataLoader,
        cur_epoch: int,
        dev: torch.device,
) -> float:
    # training loop description
    train_model.train()
    train_loss = 0.0
    # iterate over dataset
    for i, data in enumerate(train_loaded, 1):
        states, ctrls = to_device(dev, *data)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward pass and loss calculation
        p_ctrls = train_model(states)
        p_ctrls = torch.squeeze(p_ctrls)
        loss = loss_fn(p_ctrls, ctrls)

        # backward pass
        loss.backward()

        # optimizer run
        optimizer.step()

        train_loss += loss.item()

    print(f'Epoch {cur_epoch}, Loss: {train_loss / len(train_loaded)}')
    return train_loss / len(train_loaded)


def val_one_epoch(
        train_model: nn.Module,
        loss_fn: torch.nn.Module,
        val_loader: DataLoader,
        cur_epoch: int,
        best: float,
        dev: torch.device,
        ckpt_path: Optional[str],
) -> tuple[float, float]:
    # validation
    val_loss = 0.0
    with torch.no_grad():
        train_model.eval()  # evaluation mode
        for i, data in enumerate(val_loader, 1):
            states, ctrls = to_device(dev, *data)

            p_ctrls = train_model(states)
            p_ctrls = torch.squeeze(p_ctrls)
            val_loss += loss_fn(p_ctrls, ctrls).item()

        print(f'Validation {cur_epoch}, Loss: {val_loss / len(val_loader)}')

        if val_loss / len(val_loader) < best and ckpt_path:
            torch.save(train_model.state_dict(), ckpt_path)
            best = val_loss / len(val_loader)

    return best, val_loss / len(val_loader)


class BaseAiCtrl(nn.Module):
    def __init__(self):
        super(BaseAiCtrl, self).__init__()

    def train_model(self,
                    train_loader: DataLoader,
                    val_loader: Optional[DataLoader],
                    epochs: int,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    ckpt_path: Optional[str] = "best.pt") -> Union[tuple[list[float], list[float]], list[float]]:
        """
        Train the AI control model.

        Parameters:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (Optional[DataLoader]): DataLoader for validation data. Can be None.
            epochs (int): Number of training epochs.
            optimizer (Optional[torch.optim.Optimizer], optional): Optimizer for training.
            ckpt_path (str, optional): Path to save the best model checkpoint. If None, no checkpoint will be saved

        Returns:
            Union[tuple[list[float], list[float]], list[float]]: Training losses. If validation data is provided,
                returns a tuple containing training and validation losses.
        """
        optimizer = optimizer or torch.optim.Adam(self.parameters())
        device: torch.device = self._dummy_param.device
        loss_fn = nn.MSELoss()

        train_losses = []
        val_losses = []
        best = float('inf')

        for epoch in range(epochs):
            train_loss = train_one_epoch(
                self,
                optimizer,
                loss_fn,
                train_loader,
                epoch + 1,
                device,
            )
            train_losses.append(train_loss)

            if val_loader:
                best, val_loss = val_one_epoch(
                    self,
                    loss_fn,
                    val_loader,
                    epoch + 1,
                    best,
                    device,
                    ckpt_path,
                )
                val_losses.append(val_loss)
                continue

            # Save model by train loss
            if best > train_loss and ckpt_path:
                torch.save(self.state_dict(), ckpt_path)
                best = train_loss

        if val_loader:
            return train_losses, val_losses
        return train_losses

    def make_predictions(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Make predictions using the AI control model.

        Parameters:
            x (torch.Tensor): Input tensor for prediction.
            **kwargs (Any): Additional parameters.

        Returns:
            torch.Tensor: Model predictions.
        """
        x_dev = 'cuda' if x.get_device() > -1 else 'cpu'
        m_dev: torch.device = self._dummy_param.device

        # To model device
        if x_dev != m_dev.type:
            x = x.to(m_dev)

        result = self._make_predictions(x, **kwargs)

        # Back to initial device
        if x_dev != m_dev.type:
            result = result.to(x_dev)

        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AI Control model.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        raise NotImplementedError("Should be implemented by child classes")

    def _make_predictions(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Make predictions using the AI control model.

        Parameters:
            x (torch.Tensor): Input tensor for prediction.
            **kwargs (Any): Additional parameters.

        Returns:
            torch.Tensor: Model predictions.
        """
        raise NotImplementedError("Should be implemented by child classes")
