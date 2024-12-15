import torch

from typing import Union, Optional
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def write_model_info(ckpt_path: Union[str, Path], train_model: nn.Module, loss: float):
    ckpt_path = Path(ckpt_path)
    model_name = ckpt_path.stem
    info_path = ckpt_path.parent / f"{model_name}.txt"
    model_info = str(train_model)
    with open(info_path, "w") as file:
        file.write(f"loss: {loss}\n\n")
        file.write(model_info)


def to_device(
    _device: torch.device, *tensors: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    return tuple(t.to(_device) for t in tensors)


def train_one_epoch(
    train_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    train_loader: DataLoader,
    dev: torch.device,
    epoch: int,
) -> float:
    # training loop description
    train_model.train()
    train_loss = 0.0
    # iterate over dataset
    with tqdm(enumerate(train_loader, 1), unit="batch", total=len(train_loader)) as bar:
        for i, data in bar:
            bar.set_description(f"Training, epoch {epoch}")
            states, ctrls = to_device(dev, *data)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass and loss calculation
            p_ctrls = train_model(states)
            p_ctrls = torch.squeeze(p_ctrls)
            ctrls = torch.squeeze(ctrls)
            loss = loss_fn(p_ctrls, ctrls)

            # backward pass
            loss.backward()

            # optimizer run
            optimizer.step()

            train_loss += loss.item()
            bar.set_postfix(loss=train_loss / i)

    return train_loss / len(train_loader)


def val_one_epoch(
    train_model: nn.Module,
    loss_fn: torch.nn.Module,
    loader: DataLoader,
    dev: torch.device,
    epoch: int,
    prefix: str = "Evaluation",
) -> float:
    # validation
    mean_loss = 0.0
    with torch.no_grad():
        train_model.eval()  # evaluation mode

        # Compute loss
        val_loss = 0.0
        with tqdm(enumerate(loader, 1), unit="batch", total=len(loader)) as bar:
            for i, data in bar:
                bar.set_description(f"{prefix}, epoch {epoch}")
                states, ctrls = to_device(dev, *data)

                p_ctrls = train_model(states)
                p_ctrls = torch.squeeze(p_ctrls)
                ctrls = torch.squeeze(ctrls)
                val_loss += loss_fn(p_ctrls, ctrls).item()
                bar.set_postfix(loss=val_loss / i)

        mean_loss = val_loss / len(loader)

    return mean_loss


class BaseAiCtrl(nn.Module):
    def __init__(self):
        super(BaseAiCtrl, self).__init__()

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        epochs: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        ckpt_path: Optional[str] = "best.pt",
    ) -> Union[tuple[list[float], list[float]], list[float]]:
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
        best = float("inf")

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(
                self,
                optimizer,
                loss_fn,
                train_loader,
                device,
                epoch,
            )
            train_losses.append(train_loss)

            if val_loader:
                val_loss = val_one_epoch(
                    self, loss_fn, val_loader, device, epoch, prefix="Validation"
                )
                val_losses.append(val_loss)
            else:
                val_loss = None

            # Save model by loss
            cur_loss = val_loss if val_loss is not None else train_loss
            if cur_loss < best and ckpt_path:
                torch.save(self.state_dict(), ckpt_path)
                best = cur_loss
                write_model_info(ckpt_path, self, best)

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
        x_dev = "cuda" if x.get_device() > -1 else "cpu"
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
