# define recorder for simulation data

from typing import List
import numpy as np
from copy import deepcopy as copy
import mujoco
import pandas as pd


class Recorder:
    def __init__(
        self,
        model: mujoco.MjModel,
        data_attrs: List[str] = ["time", "qpos", "qvel", "qacc", "ctrl"],
    ):
        self.data_attrs = data_attrs

        self.history = dict()
        for attr in data_attrs:
            self.history[attr] = []

        self.sensors_names = []
        for i in range(model.nsensor):
            self.sensors_names.append(model.sensor(i).name)

        for sensor_name in self.sensors_names:
            self.history[sensor_name] = []

    def record(self, data):
        for attr in self.data_attrs:
            self.history[attr].append(copy(getattr(data, attr)))
        for sensor_name in self.sensors_names:
            self.history[sensor_name].append(copy(data.sensor(sensor_name).data))

    def save_df(self, filepath):
        # dataframe should contain all data attributes and sensor data
        df = pd.DataFrame(self.history)
        df.to_csv(filepath)

    def save(self, filepath):
        # history is dict which can be saved as pickle
        for k, v in self.history.items():
            self.history[k] = np.array(v)

        import pickle

        with open(filepath, "wb") as f:
            pickle.dump(self.history, f)
