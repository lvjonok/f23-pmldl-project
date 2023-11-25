import numpy as np
import pandas as pd
import torch
torch.manual_seed(420)
from typing import Union, Tuple


class PhysicsStatesDataset(torch.utils.data.Dataset):
    '''Dataset class for reading data from the simulations'''

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe
        self.n_coordinates = 2
        self.states, self.n_coordinates = self.preprocess(self.dataframe)

    def read_transform(self, entity: str) -> list:
        '''
        This methods allows to convert string of numbers separated by spaces into a list of float values

        Args:
            entity (str): a column value to be transformed into a list of floats

        Returns:
            list: a list of float values converted from the string
        '''
        entity = entity[1:-1]
        entities = entity.split(" ")
        numbers = []
        for number in entities:
          if number != '':
            numbers.append(float(number))
        return numbers

    def listify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        This methods converts values in the initial csv file to list of float numbers for each column

        Args:
            df (pd.DataFrame): a dataframe to take values from

        Returns:
            pd.DataFram: a listified dataframe
        '''
        new_df = df.copy()
        columns = ['qpos', 'qvel', 'qacc', 'ctrl']
        for column in columns:
            new_df[column] = new_df[column].apply(self.read_transform)
        return new_df

    def flatten_row(self, df_row)  -> list:
        ''' 
        A simple function to flatten all the states

        Args:
            df_row : a dataframe row consisting of several lists of floats

        Returns:
            list: flattened state
        '''
        states = np.array(df_row.values.flatten().tolist()).flatten()
        return states

    def preprocess(self, df: pd.DataFrame) -> Tuple[list, int]:
        ''' 
        Function that preprocess a dataframe from the experiment output format to list of states

        Args:
            df: a dataframe to be processed

        Returns:
            list: list of flattened states
        '''
        df = self.listify_dataframe(df)
        df= df.drop(['id', 'time'], axis=1)
        n_coordinates = len(df['ctrl'][0])
        states = np.array([self.flatten_row(df.iloc[i, :]) for i in range(len(df))]).astype(np.float32)
        return states, n_coordinates


    def join_csv(self, csv: Union[str, pd.DataFrame]) -> bool:
        '''
        Function for joining another csv file to this dataset

        Args:
            csv (str or pd.DataFrame): a path to the csv file to be added or the dataframe itself

        Returns:
            bool: False in case of mismatching dimensions. True otherwise
        '''
        if type(csv) == str:
          csv = pd.read_csv(csv)
        csv_states, csv_coords = self.preprocess(csv)
        if csv_coords != self.n_coordinates:
          return False
        self.states = np.concatenate((self.states, csv_states))
        return True
        

    def _get_state(self, index: int) -> list:
        # retrieves a state from dataset by index
        return self.states[index]

    def __getitem__(self, index) -> tuple[list, list]:
        return self._get_state(index)

    def __len__(self) -> int:
        return len(self.states)