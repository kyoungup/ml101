import unittest
from pathlib import Path
import pandas as pd
import numpy as np
from ml101.serialize import Data, Stream
from ml101.preprocess import DataFilter


CWD = Path(__file__).parent
TEMP = CWD / 'temp'
TEMP.mkdir(exist_ok=True, parents=True)


class TestInsertion(unittest.TestCase):
    def setUp(self) -> None:
        self.data = Data(
            pd.DataFrame(
                [[np.nan, 2, np.nan, 0],
                    [3, 4, np.nan, 1],
                    [np.nan, np.nan, np.nan, 5],
                    [np.nan, 3, np.nan, 4]],
                    columns=list('ABCD'), dtype=float)
        )
        self.filter = DataFilter(self.data)

    def tearDown(self) -> None:
        pass

    def test_fill_na(self):
        colvals = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        self.filter.fill_na(colvals)
        gt = Data(
            pd.DataFrame(
                [[0, 2, 2, 0],
                [3, 4, 2, 1],
                [0, 1, 2, 5],
                [0, 3, 2, 4]],
                columns=list('ABCD'), dtype=float)
        )
        assert self.filter._data.equals(gt)

    def test_fill_na_tuple(self):
        colvals = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        cols = [*colvals]
        vals = [colvals[col] for col in cols]
        self.filter.fill_na((cols, vals))
        gt = Data(
            pd.DataFrame(
                [[0, 2, 2, 0],
                [3, 4, 2, 1],
                [0, 1, 2, 5],
                [0, 3, 2, 4]],
                columns=list('ABCD'), dtype=float)
        )
        assert self.filter._data.equals(gt)

    def test_fill_median(self):
        cols = ['A', 'B']
        self.filter.fill_median()
        gt = Data(
            pd.DataFrame(
                [[3, 2, np.nan, 0],
                [3, 4, np.nan, 1],
                [3, 3, np.nan, 5],
                [3, 3, np.nan, 4]],
                columns=list('ABCD'), dtype=float)
        )
        assert self.filter._data.equals(gt)

    def test_fill_mean(self):
        self.filter.fill_mean()
        gt = Data(
            pd.DataFrame(
                [[3, 2, np.nan, 0],
                [3, 4, np.nan, 1],
                [3, 3, np.nan, 5],
                [3, 3, np.nan, 4]],
                columns=list('ABCD'), dtype=float)
        )
        assert self.filter._data.equals(gt)

    def test_fill_along_backward(self):
        self.filter.fill_along(method='bfill')
        gt = Data(
            pd.DataFrame(
                [[3, 2, np.nan, 0],
                [3, 4, np.nan, 1],
                [np.nan, 3, np.nan, 5],
                [np.nan, 3, np.nan, 4]],
                columns=list('ABCD'), dtype=float)
        )
        assert self.filter._data.equals(gt)


class TestRemoval(unittest.TestCase):
    def setUp(self) -> None:
        self.data = Data(
            pd.DataFrame({"name": ['Alfred', 'Batman', 'Catwoman'],
                "toy": [np.nan, 'Batmobile', 'Bullwhip'],
                "born": [pd.NaT, pd.Timestamp("1940-04-25"), pd.NaT],
                'CONST': [0, 0, 0]})
        )
        self.filter = DataFilter(self.data)

        df_outlier = Stream(CWD / 'data' / 'boston.csv').read()
        self.filter_outlier = DataFilter(df_outlier)

    def tearDown(self) -> None:
        pass

    def test_drop_na(self):
        self.filter.drop_na(how='any')
        gt = Data(
            pd.DataFrame(
                [['Batman', 'Batmobile', pd.Timestamp("1940-04-25"), 0]],
                columns=['name', 'toy', 'born', 'CONST'])
        )
        assert self.filter._data.equals(gt)

    def test_drop_const(self):
        self.filter.drop_const()
        gt = Data(
           pd.DataFrame({"name": ['Alfred', 'Batman', 'Catwoman'],
                "toy": [np.nan, 'Batmobile', 'Bullwhip'],
                "born": [pd.NaT, pd.Timestamp("1940-04-25"), pd.NaT]})
        )
        assert self.filter._data.equals(gt)

    def test_drop_columns(self):
        remove_cols = ['toy', 'CONST']
        self.filter.drop_columns(remove_cols)
        gt = Data(
            pd.DataFrame({"name": ['Alfred', 'Batman', 'Catwoman'],
                "born": [pd.NaT, pd.Timestamp("1940-04-25"), pd.NaT]})
        )
        assert self.filter._data.equals(gt)

    def test_drop_rows(self):
        remove_rows = [0, 2]
        self.filter.drop_rows(remove_rows)
        gt = Data(
            pd.DataFrame({"name": ['Batman'],
                "toy": ['Batmobile'],
                "born": [pd.Timestamp("1940-04-25")],
                'CONST': [0]})
        )
        assert self.filter._data.equals(gt)

    def test_drop_outliers(self):
        self.filter_outlier.drop_outliers()
        assert self.filter_outlier._data.shape == (274, 13)
    # TODO: drawing scatter plot with outliers


class TestConversion(unittest.TestCase):
    def setUp(self) -> None:
        self.data = Data(
            pd.DataFrame(
                [[0, 0], [0, 0], [1, 1], [1, 1]],
                columns=list('AB'), dtype=float)
        )
        self.filter = DataFilter(self.data)

    def tearDown(self) -> None:
        pass

    def test_scale_standard(self):
        self.filter.scale()
        gt = Data(
            pd.DataFrame(
                [[-1., -1.], [-1., -1.], [ 1.,  1.], [ 1.,  1.]],
                columns=list('AB'), dtype=float)
        )
        assert self.filter._data.equals(gt)

    def test_scale_standard_exclude_cols(self):
        self.filter.scale(except_cols=['B'])
        gt = Data(
            pd.DataFrame(
                [[-1., 0], [-1., 0], [ 1.,  1.], [ 1.,  1.]],
                columns=list('AB'), dtype=float)
        )
        assert self.filter._data.equals(gt)

    def test_scale_minmax(self):
        self.data_minmax = Data(
            pd.DataFrame(
                [[-1, 2], [-0.5, 6], [0, 10], [1, 18]],
                columns=list('AB'), dtype=float)
        )
        self.filter_minmax = DataFilter(self.data_minmax)

        self.filter_minmax.scale(method=DataFilter.SCALE_MINMAX)
        gt = Data(
            pd.DataFrame(
                [[0., 0.], [0.25, 0.25], [0.5,  0.5], [1., 1.]],
                columns=list('AB'), dtype=float)
        )
        assert self.filter_minmax._data.equals(gt)

    def test_onehot(self):
        self.data_onehot = Data(
            pd.DataFrame(
                [['Male', 1], ['Female', 3], ['Female', 2]],
                columns=['Gender','Age'])
        )
        self.filter_onehot = DataFilter(self.data_onehot)

        self.filter_onehot.onehot()
        gt = Data(
            pd.DataFrame(
                [[0., 1., 1., 0., 0.],
                [1., 0., 0., 0., 1.],
                [1., 0., 0., 1., 0.]],
                )
        )
        assert (self.filter_onehot._data.dataframe.values == gt.dataframe.values).all()


class TestConversionForTimeSeries(unittest.TestCase):
    def setUp(self) -> None:
        filepath = CWD / 'data' / 'stream_sample.tsv'
        self.reader = Stream(filepath)
        self.data = self.reader.read(pos_header=4)

        self.prep = DataFilter(self.data.dataframe)

        self.sr1 = [0,0,1,1,1,0,0,1,1]
        self.sr2 = [1,1,0,0,0,1,1,0,0,1,1,1,0]

    def test_shift(self):
        new_data = self.prep.shift(columns=self.data.columns, move=-3)
        assert new_data._data.shape == (self.data.shape[0] - 3, self.data.shape[1] * 2)

    def test_shift_0(self):
        new_data = self.prep.shift(columns=self.data.columns, move=0)
        assert new_data._data.shape == (self.data.shape[0], self.data.shape[1])

    def test_period_countup(self):
        gt_ex1 = [0, 0, 1, 2, 3, 0, 0, 1, 2]
        gt_ex2 = [1, 2, 0, 0, 0, 1, 2, 0, 0, 1, 2, 3, 0]
        ex1 = list(self.prep.countup(self.sr1))
        ex2 = list(self.prep.countup(self.sr2))

        assert ex1 == gt_ex1
        assert ex2 == gt_ex2

    def test_period(self):
        gt_ex1 = [0, 0, 3, 3, 3, 0, 0, 2, 2]
        gt_ex2 = [2, 2, 0, 0, 0, 2, 2, 0, 0, 3, 3, 3, 0]
        ex1 = list(self.prep.period(self.sr1))
        ex2 = list(self.prep.period(self.sr2))
        assert ex1 == gt_ex1
        assert ex2 == gt_ex2

    def test_set_period(self):
        df = pd.DataFrame({'sr21': self.sr2, 'sr22': self.sr2})
        new_data = DataFilter(df)\
                    .set_period(df.columns, inplace=False)
        gt_ex2 = [2, 2, 0, 0, 0, 2, 2, 0, 0, 3, 3, 3, 0]
        df_gt = pd.DataFrame({'sr21_period': gt_ex2, 'sr22_period': gt_ex2})
        df_gt = pd.concat([df, df_gt], axis=1)
        assert df_gt.equals(new_data)
