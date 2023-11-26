import unittest
import torch

from tests.test_utils import get_data_loaders_from_mixed_dataset
from src.ai_models import FullyConnectedCtrl


class TestFullyConnected(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(402)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.epochs = 100

        self.train_dataloader, self.val_dataloader = get_data_loaders_from_mixed_dataset(
            n_previous_states=None,
            split_proportion=0.9,
            batch_size=100,
        )

        self.model = FullyConnectedCtrl(
            nv=1,
            ctrl=1,
            linear_layers=3,
            linear_hidden_size=100,
        ).to(device)

    def test_train_shape(self):
        states, ctrl = next(iter(self.train_dataloader))
        assert tuple(states.shape) == (100, 3)
        assert tuple(ctrl.shape) == (100, )

    def test_val_shape(self):
        states, ctrl = next(iter(self.val_dataloader))
        assert tuple(states.shape) == (100, 3)
        assert tuple(ctrl.shape) == (100, )

    def test_fc(self):
        _, val_losses = self.model.train_model(self.train_dataloader, self.val_dataloader, self.epochs, ckpt_path=None)
        assert min(val_losses) < 65
