import unittest
from ml101.serialize import Stream
from ml101.datafile import DataFile
import ml101.utils as utils
from pathlib import Path
import pandas as pd
import numpy as np


CWD = Path(__file__).parent
TEMP = CWD / 'temp'
TEMP.mkdir(exist_ok=True, parents=True)


class TestDataFile(unittest.TestCase):
    def setUp(self) -> None:
        self.filepath = CWD / 'data' / 'random_timeseries.csv'
        self.df = DataFile(self.filepath)

    def tearDown(self):
        for file in self.monthly_files:
            if file.exists(): file.unlink()

    def test_divide_by_period(self):
        self.df.divide_by_period(date_col='time', freq='M', dst_path=TEMP)
        self.monthly_files = [
            TEMP / utils.insert2filename(self.filepath.name, suffix='_2020-01'),
            TEMP / utils.insert2filename(self.filepath.name, suffix='_2020-02'),
            TEMP / utils.insert2filename(self.filepath.name, suffix='_2020-03'),
            TEMP / utils.insert2filename(self.filepath.name, suffix='_2020-04')
        ]
        for file in self.monthly_files:
            assert file.exists()
        
