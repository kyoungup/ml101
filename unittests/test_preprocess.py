import unittest
from pathlib import Path
from numpy.lib.function_base import append
import pandas as pd
from sklearn.pipeline import Pipeline

from ml101.serialize import Data, StreamFactory
from ml101.preprocess import Shift, SetPeriod, Preprocessor

CWD = Path(__file__).parent


class TestPreprocess(unittest.TestCase):
    def setUp(self) -> None:
        filepath = CWD / 'data' / 'sample.tsv'
        self.reader = StreamFactory.open(filepath)
        self.option = {'header': 4}
        self.data = self.reader.read(**self.option)
        self.col_Y = -2
        self.gt_period = [1, 0, 0, 4, 4, 4, 4, 0, 0, 6, 6, 6, 6, 6, 6, 0, 0, 2, 2]

    def tearDown(self) -> None:
        pass

    def test_Shift(self):
        workflow = Pipeline([
            ('shift', Shift(move=-3, columns=self.data.columns))
        ])
        new_data = workflow.fit_transform(self.data.dataframe)
        assert new_data.shape == (self.data.shape[0] - 3, self.data.shape[1] * 2)

    def test_SetPeriod(self):
        col_new = 'time'
        gt_data = pd.DataFrame(self.gt_period, columns=[col_new])
        workflow = Pipeline([
            ('period', SetPeriod(columns=[self.data.columns[self.col_Y]], outs=[col_new], append=False))
        ])
        new_data = workflow.fit_transform(self.data.dataframe)
        assert new_data[col_new].equals(gt_data[col_new])


class TestPreprocessForTimeSeries(unittest.TestCase):
    def setUp(self) -> None:
        # set 2 for shift()
        self.df = pd.DataFrame()
        self.df['Xa'] = list(range(10))
        self.df['Xb'] = [2 * x for x in range(10)]
        self.df['Y'] = [5 * x for x in range(10)]

        self.sr1 = [0,0,1,1,1,0,0,1,1]
        self.sr2 = [1,1,0,0,0,1,1,0,0,1,1,1,0]

    def test_SetPeriod_countup(self):
        gt_ex1 = [0, 0, 1, 2, 3, 0, 0, 1, 2]
        gt_ex2 = [1, 2, 0, 0, 0, 1, 2, 0, 0, 1, 3, 3, 0]
        ex1 = list(SetPeriod.countup(self.sr1))
        ex2 = list(SetPeriod.countup(self.sr2))

        assert ex1 == gt_ex1
        assert ex2 == gt_ex2

    def test_SetPeriod_period(self):
        gt_ex1 = [0, 0, 3, 3, 3, 0, 0, 2, 2]
        gt_ex2 = [2, 2, 0, 0, 0, 2, 2, 0, 0, 3, 3, 3, 0]
        ex1 = list(SetPeriod.countup(self.sr1))
        ex2 = list(SetPeriod.countup(self.sr2))

        assert ex1 == gt_ex1
        assert ex2 == gt_ex2


class TestDropFunctions(unittest.TestCase):
    def setUp(self) -> None:
        filepath = CWD / 'data' / 'NCC_sample.xlsx'
        self.readstream = StreamFactory.open(filepath)

    def tearDown(self) -> None:
        pass

    def test_dropna(self):
        self.data = self.readstream.read()
        prc = Preprocessor(self.data)
        sum_of_nan = prc.data.dataframe.isnull().sum().sum()
        print('\nsum of nan before dropna():', sum_of_nan)
        sum_of_nan = prc.dropna().data.dataframe.isnull().sum().sum()
        print('\nsum of nan after dropna():', sum_of_nan)
        assert sum_of_nan == 0

    def test_dropconst(self):
        self.data = self.readstream.read()
        prc = Preprocessor(self.data)
        prc.data.dataframe['RM'] = 1
        assert 'RM' not in prc.dropconst().data.daatframe.columns

    def test_drop_columns(self):
        self.data = self.readstream.read()
        remove_cols = self.data.columns[2:4]
        prc = Preprocessor(self.data)
        prc.drop_columns(remove_cols)
        not_removed = 0
        for i in range(len(remove_cols)):
            if remove_cols[i] in prc.data.dataframe.columns:
                not_removed += 1
        assert not_removed == 0

    def test_drop_rows(self):
        self.data = self.readstream.read()
        remove_rows = [1,2,3]
        prc = Preprocessor(self.data)
        prc.drop_rows(remove_rows)
        not_removed = 0
        for i in range(len(remove_rows)):
            if remove_rows[i] in prc.data.dataframe.index:
                not_removed += 1
        assert not_removed == 0

    def test_drop_outliers(self):
        self.data = self.readstream.read()
        raw_data_size = self.data.shape[0]
        prc = Preprocessor(self.data)
        column = ['N1_TI2177B']
        drop_region = 'lower' # both, upper
        prc.drop_outliers(column, drop_region)

        assert prc.data.dataframe.shape[0] < raw_data_size
