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
        ckpt_path: str,
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

        if val_loss / len(val_loader) < best:
            torch.save(train_model.state_dict(), ckpt_path)
            best = val_loss / len(val_loader)

    return best, val_loss / len(val_loader)


class RnnCtrl(nn.Module):
    def __init__(self,
                 nv: int,
                 ctrl: int,
                 rnn_hidden_size: int,
                 rnn_layers: int,
                 linear_layers: int,
                 linear_hidden_size: int,
                 is_bidirectional: bool,
                 rnn_cell_type: Union[type[nn.LSTM], type[nn.GRU]] = nn.GRU,
                 rnn_dropout: float = 0,
                 linear_dropout: float = 0,
                 linear_activation: Optional[nn.Module] = None,
                 rnn_initials: Optional[list[torch.Tensor]] = None):
        """
        Initialize the RnnCtrl model.

        Parameters:
            nv (int): Number of generalized velocities in the input.
            ctrl (int): Number of generalized controls.
            rnn_hidden_size (int): Number of features in the hidden state of the RNN.
            rnn_layers (int): Number of layers in the RNN.
            linear_layers (int): Number of linear layers in the fully connected part.
            linear_hidden_size (int): Number of features in the hidden layers of the fully connected part.
            is_bidirectional (bool): Whether the RNN is bidirectional or not.
            rnn_cell_type (Union[type[nn.LSTM], type[nn.GRU]], optional): Type of RNN cell (LSTM or GRU).
            rnn_dropout (float, optional): Dropout rate for the RNN layers.
            linear_dropout (float, optional): Dropout rate for the linear layers.
            linear_activation (Optional[nn.Module], optional): Activation function for the linear layers.
            rnn_initials (Optional[list[torch.Tensor]], optional): Initial state for the RNN layers.

        Raises:
            ValueError: If an unsupported RNN cell type is provided.
        """

        super(RnnCtrl, self).__init__()
        self.rnn_cell_type = rnn_cell_type
        self._d = 2 if is_bidirectional else 1
        self.linear_activation = linear_activation or nn.Sigmoid()
        self._linear_layers = linear_layers
        self._linear_hidden_size = linear_hidden_size
        self.nv = nv
        self.rnn_initials = rnn_initials or []

        if not (rnn_cell_type is nn.GRU or rnn_cell_type is nn.LSTM):
            raise ValueError('Only GRU and LSTM cell types are supported!')

        # Rnn part
        self.rnn = rnn_cell_type(
            input_size=3 * nv,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            bidirectional=is_bidirectional,
            dropout=rnn_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=linear_dropout)
        # Linear part
        self.linear = nn.Sequential(
            # Hidden linear layers
            *[
                nn.Sequential(
                    nn.Linear(self._d * rnn_hidden_size, linear_hidden_size)
                    if i == 0 else
                    nn.Linear(linear_hidden_size, linear_hidden_size),
                    self.linear_activation
                )
                for i in range(linear_layers - 1)
            ],
            # Last layer
            nn.Linear(linear_hidden_size, ctrl)
            if linear_layers > 1 else
            nn.Linear(self._d * rnn_hidden_size, ctrl)
        )
        self._dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RnnCtrl model.

        Parameters:
            x (torch.Tensor): Input tensor of shape (L, 3 * nv) or (N, L, 3 * nv).

        Returns:
            torch.Tensor: Output tensor.
        """
        if x.shape[-1] != 3 * self.nv:
            raise ValueError('Input tensor should have shape (L, 3 * nv) or (N, L, 3 * nv)')

        if self.rnn_cell_type is nn.GRU:
            rnn_out, _ = self.rnn(x, *self.rnn_initials)
        else:
            rnn_out, (_, _) = self.rnn(x, *self.rnn_initials)

        if len(rnn_out.shape) == 2:
            rnn_out = rnn_out[-1]  # unbatched input
        else:
            rnn_out = rnn_out[:, -1]

        return self.dropout(self.linear(rnn_out))

    def train_model(self,
                    train_loader: DataLoader,
                    val_loader: Optional[DataLoader],
                    epochs: int,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    ckpt_path: str = "best.pt") -> Union[tuple[list[float], list[float]], list[float]]:
        """
        Train the RnnCtrl model.

        Parameters:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (Optional[DataLoader]): DataLoader for validation data. Can be None.
            epochs (int): Number of training epochs.
            optimizer (Optional[torch.optim.Optimizer], optional): Optimizer for training.
            ckpt_path (str, optional): Path to save the best model checkpoint.

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
            if best > train_loss:
                torch.save(self.state_dict(), ckpt_path)
                best = train_loss

        if val_loader:
            return train_losses, val_losses
        return train_losses

    def make_predictions(self, x: torch.Tensor, prev_steps: int = 1) -> torch.Tensor:
        """
        Make predictions using the RnnCtrl model.

        Parameters:
            x (torch.Tensor): Input tensor for prediction.
            prev_steps (int, optional): Number of previous steps to consider for each prediction.

        Returns:
            torch.Tensor: Model predictions.
        """
        x_dev = 'cuda' if x.get_device() > -1 else 'cpu'
        m_dev: torch.device = self._dummy_param.device

        # To model device
        if x_dev != m_dev.type:
            x = x.to(m_dev)

        if prev_steps == 1:
            x = x.unsqueeze(-2)
        else:
            x = x.unfold(0, prev_steps, 1).transpose(2, 1)

        with torch.no_grad():
            result = self(x, *self.rnn_initials)

        # Back to initial device
        if x_dev != m_dev.type:
            result = result.to(x_dev)

        return result


def test_rnn_ctrl():
    import os
    import pandas as pd

    from ast import literal_eval
    from torch.utils.data import TensorDataset, random_split
    from matplotlib import pyplot as plt

    base = '../data/interim'
    data_path = 'pendulum/mixed1.csv'
    best_ckpt_path = "best_rnn.pt"
    full_path = os.path.join(base, data_path)
    states_columns = ['qpos', 'qvel', 'qacc']
    num_of_prev_states = 5
    num_epochs = 100
    split_proportion = 0.9
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    raw = pd.read_csv(
        full_path,
        converters={
            'qpos': literal_eval,
            'qvel': literal_eval,
            'qacc': literal_eval,
            'ctrl': literal_eval,
        }
    )
    for cln in ['qpos', 'qvel', 'qacc', 'ctrl']:
        raw[cln] = raw[cln].apply(lambda x: x[0])

    first_half = raw.iloc[:10001]
    second_half = raw.iloc[10001:].reset_index().drop('index', axis=1)

    first_states = first_half[states_columns].values
    first_ctrl = first_half['ctrl'].values
    second_states = second_half[states_columns].values
    second_ctrl = second_half['ctrl'].values

    first_states = torch.tensor(first_states, dtype=torch.float32)
    first_ctrl = torch.tensor(first_ctrl, dtype=torch.float32)[num_of_prev_states - 1:]
    second_states = torch.tensor(second_states, dtype=torch.float32)
    second_ctrl = torch.tensor(second_ctrl, dtype=torch.float32)[num_of_prev_states - 1:]

    first_states = first_states.unfold(0, num_of_prev_states, 1).transpose(2, 1)
    second_states = second_states.unfold(0, num_of_prev_states, 1).transpose(2, 1)

    all_states = torch.cat((first_states, second_states))
    all_ctrls = torch.cat((first_ctrl, second_ctrl))

    all_data = TensorDataset(all_states, all_ctrls)
    size = int(len(all_data) * split_proportion)
    train_dataset, val_dataset = random_split(all_data, [size, len(all_data) - size])

    batch_size = 50
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    model = RnnCtrl(
        nv=1,
        ctrl=1,
        rnn_hidden_size=10,
        rnn_layers=2,
        linear_layers=2,
        linear_hidden_size=100,
        is_bidirectional=False,
    ).to(device)
    train_losses, val_losses = model.train_model(train_dataloader, val_dataloader, num_epochs, ckpt_path=best_ckpt_path)

    epochs_seq = list(range(1, num_epochs + 1))

    plt.cla()
    plt.title('Loss history')
    plt.plot(epochs_seq, train_losses, color='b', label='Train loss')
    plt.plot(epochs_seq, val_losses, color='r', label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    test_rnn_ctrl()
