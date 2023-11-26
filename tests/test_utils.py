import os

from typing import Optional
from torch.utils.data import random_split, DataLoader
from src.physics_dataset import PhysicsStatesDataset


def get_data_loaders_from_mixed_dataset(
        n_previous_states: Optional[int] = 5,
        split_proportion: float = 0.9,
        batch_size: int = 100,
        base: str = '../data/interim',
        data_path: str = 'pendulum/mixed1.csv',
) -> tuple[DataLoader, DataLoader]:
    full_path = os.path.join(base, data_path)

    raw = PhysicsStatesDataset.load_dataframe(full_path)
    first_half = raw.iloc[:10001]
    second_half = raw.iloc[10001:].reset_index().drop('index', axis=1)

    dataset = PhysicsStatesDataset(first_half, n_previous_states)
    dataset.join_dataframe(second_half)

    size = int(len(dataset) * split_proportion)
    train_dataset, val_dataset = random_split(dataset, [size, len(dataset) - size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    return train_dataloader, val_dataloader
