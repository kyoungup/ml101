import pandas as pd
import numpy as np

class TableData(pd.DataFrame):
    def __init__(self):
        super().__init__()
    

class ListData(list):
    def __init__(self):
        super().__init__()


class Data:
    def __init__(self, data=None):
        if isinstance(data, pd.DataFrame):
            self._dataframe = data
        elif isinstance(data, np.ndarray):
            self._dataframe = pd.DataFrame(data)
        else:
            self._dataframe = pd.DataFrame()

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self._dataframe._values, dtype=dtype)

class Datap(pd.DataFrame):
    def __init__(self, data=None):
        super().__init__(data)