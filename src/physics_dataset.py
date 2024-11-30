import pandas as pd
import torch
import numpy as np

from typing import Optional, Union, Callable, Sequence
from pathlib import Path
from ast import literal_eval
from torch.utils.data import Dataset


class PhysicsStatesDataset(Dataset):
    """Dataset class for reading data from the simulations"""

    def __init__(self,
                 dataframe_or_path: Union[pd.DataFrame, str, Path],
                 group_by_n_previous_states: Optional[int] = None,
                 state_names: Optional[list[str]] = None,
                 ctrl_name: Optional[str] = None,
                 ctrl_noise: Optional[Callable[[Sequence[int]], torch.Tensor]] = None,
                 state_noise: Optional[Callable[[Sequence[int]], torch.Tensor]] = None,
                 shuffle: bool = True,
                 dtype: torch.dtype = torch.float32,):
        self.group_by_n_previous_states = group_by_n_previous_states
        self.state_names = state_names or ['qpos', 'qvel', 'qacc']
        self.ctrl_name = ctrl_name or 'ctrl'
        self.dtype = dtype
        self.shuffle = shuffle
        self.state_noise = state_noise
        self.ctrl_noise = ctrl_noise
        self.dataframe = self.load_dataframe(dataframe_or_path, state_names, ctrl_name, shuffle)
        self.states, self.ctrls = self.preprocess(self.dataframe)

    @staticmethod
    def load_dataframe(dataframe_or_path: Union[pd.DataFrame, str, Path],
                       state_names: Optional[list[str]] = None,
                       ctrl_name: Optional[str] = None,
                       shuffle: bool = True) -> pd.DataFrame:
        if isinstance(dataframe_or_path, pd.DataFrame):
            return dataframe_or_path

        _state_names: list[str] = state_names or ['qpos', 'qvel', 'qacc']
        _ctrl_name: str = ctrl_name or 'ctrl'
        converters: dict[str, Callable] = {
            i: lambda x: np.array(x[1:-1].split(), dtype="float")
            for i in _state_names + [_ctrl_name]
        }
        df = pd.read_csv(
            dataframe_or_path,
            converters=converters,
            index_col=0,
        )
        if shuffle:
            df = df.sample(frac=1)
            df = df.reset_index(drop=True)
        return df

    def preprocess(self, df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        states = torch.tensor(np.array([[*qpos, *qvel, *qacc] for qpos, qvel, qacc in df[self.state_names].values]))
        states = torch.squeeze(states)
        states = states.type(self.dtype)

        ctrl = torch.tensor(np.array([i for i in df[self.ctrl_name].values]))
        ctrl = torch.squeeze(ctrl)
        ctrl = ctrl.type(self.dtype)

        if self.group_by_n_previous_states:
            states = states.unfold(0, self.group_by_n_previous_states, 1).transpose(2, 1)
            ctrl = ctrl[self.group_by_n_previous_states - 1:]

        if self.state_noise:
            states += self.state_noise(states.shape)
        if self.ctrl_noise:
            ctrl += self.ctrl_noise(ctrl.shape)

        return states, ctrl

    def join_dataframe(self, dataframe_or_path: Union[pd.DataFrame, str, Path]):
        dataframe = self.load_dataframe(dataframe_or_path, self.state_names, self.ctrl_name, self.shuffle)
        self.dataframe = pd.concat([self.dataframe, dataframe])
        self.dataframe = self.dataframe.reset_index(drop=True)

        new_states, new_ctrls = self.preprocess(dataframe)
        self.states = torch.concat([self.states, new_states])
        self.ctrls = torch.concat([self.ctrls, new_ctrls])

    def normalize_angle(self, axis: int):
        angles = self.states[:, axis]
        k = (angles / (2 * torch.pi)).type(torch.int32).type(torch.int32)
        rots = 2 * torch.pi * k
        angles -= rots
        self.states[:, axis] = angles

    def reshuffle(self):
        self.dataframe = self.dataframe.sample(frac=1)
        self.dataframe = self.dataframe.reset_index(drop=True)
        self.states, self.ctrls = self.preprocess(self.dataframe)

    def _get_state(self, index: int) -> torch.Tensor:
        # retrieves a state from dataset by index
        return self.states[index]

    def _get_ctrl(self, index: int) -> torch.Tensor:
        # retrieves a control from dataset by index
        return self.ctrls[index]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._get_state(index), self._get_ctrl(index)

    def __len__(self) -> int:
        return len(self.states)
