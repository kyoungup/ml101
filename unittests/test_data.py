import unittest
from ml101.serialize import Stream
from ml101.data import Data, IndexList
from pathlib import Path
import pandas as pd
import numpy as np


CWD = Path(__file__).parent
TEMP = CWD / 'temp'
TEMP.mkdir(exist_ok=True, parents=True)


class TestData(unittest.TestCase):
    def setUp(self) -> None:
        filepath = CWD / 'data' / 'stream_sample.tsv'
        self.stream = Stream(filepath)
        self.data = self.stream.read(pos_header=4)

    def tearDown(self):
        pass

    def test_assign(self):
        newdata = self.data
        assert id(newdata) == id(self.data)

    def test_new_Data(self):
        newdata = Data(self.data)
        assert id(newdata) != id(self.data)

    def test_new_DataFrame(self):
        newdata = Data(self.data._dataframe)
        assert id(newdata) != id(self.data)
        assert newdata.equals(self.data)

    def test_crop(self):
        rows = [0, 1, slice(4, 8), slice(8, 9), 11]
        cols = [slice(0, 3), 4, 6]
        newdata = self.data.crop(rows, cols)
        assert newdata.shape == (2 + (8-4) + (9-8) + 1, (3-0) + 2)

    def test_crop_row(self):
        rows = [0, 1, slice(4, 8), slice(8, 9), 11]
        cols = None #[slice(0, 3), 4, 6]
        newdata = self.data.crop(rows, cols)
        assert newdata.shape == (2 + (8-4) + (9-8) + 1, self.data.shape[1])

    def test_head(self):
        assert self.data.head(10).shape[0] == 10

    def test_tail(self):
        assert self.data.tail().shape[0] == 5

    def test_columns(self):
        assert len(self.data.columns) == self.data.shape[1]

    def test_equals(self):
        data = Data({1: [10, 20], 2: [20, np.nan], 'a': ['data', 'test']})
        different_column_type = Data({1.0: [10.0, 20.0], 2.0: [20.0, np.nan], 'a': ['data', 'test']})
        assert data.equals(different_column_type)

    def test_compare_round(self):
        data = Data({1: [10, 20], 2: [20, np.nan], 'a': ['data', 'test']})
        different_column_type = Data({1.0: [10.001, 20.00], 2.0: [20.003, np.nan], 'a': ['data', 'test']})
        assert data.equals(different_column_type, ndigits=1)

    def test_na_ratio(self):
        assert (self.data.na_ratio('row') > 0).sum() == 6
    
    def test_resolve_dtypes(self):
        data = Data(
            pd.DataFrame(
                {
                    "a": pd.Series([1, 2, 3], dtype=np.dtype("int32")),
                    "b": pd.Series(["x", "y", "z"], dtype=np.dtype("O")),
                    "c": pd.Series([True, False, np.nan], dtype=np.dtype("O")),
                    "d": pd.Series(["h", "i", np.nan], dtype=np.dtype("O")),
                    "e": pd.Series([10, np.nan, 20], dtype=np.dtype("float")),
                    "f": pd.Series([np.nan, 100.5, 200], dtype=np.dtype("float")),
                    "g": pd.Series(['2021-01-01', '2021-01-02', '2021-01-03'], dtype=np.dtype("O")),
                }
            )
        )
        df_types = data.resolve_dtypes()
        gt_dtypes = pd.Series(['Int32', 'string', 'boolean', 'string',  'Int64', 'float64', 'string'], index=['a','b','c','d','e','f','g'])
        assert df_types.equals(gt_dtypes)

    def test_index_time(self):
        data = Data(
            pd.DataFrame(
                {'time': pd.date_range("2020-1-1", periods=100),
                 'value1': np.random.randn(100).cumsum(),
                 'value2': np.random.randn(100).cumsum()
                }
            )
        )
        data.time_series_on('time')
        assert data.dataframe.index.name == 'time'
        assert data.columns == ['value1', 'value2']
        assert data.index_time == 'time'
        data.time_series_off()
        assert data.dataframe.index.name is None
        assert data.columns == ['time', 'value1', 'value2']
        assert data.index_time is None


class TestIndexList(unittest.TestCase):
    def setUp(self) -> None:
        self.maxlen = 1000

        filepath = CWD / 'data' / 'stream_sample.tsv'
        self.stream = Stream(filepath)
        self.data = self.stream.read(pos_header=4)

    def tearDown(self) -> None:
        pass

    def test_idx_by_int(self):
        index = [0, 1, slice(4, 8), slice(8, 9), 11, -1]
        index_gt = [0, 1, 4, 5, 6, 7, 8, 11, self.maxlen-1]
        assert IndexList(index, self.maxlen).list == index_gt

    def test_idx_by_excel(self):
        index = ['A', 'C', slice('D', 'F'), 'J']
        index_gt = [0, 2, 3, 4, 9]
        assert IndexList(index, self.maxlen).list == index_gt

    def test_idx_by_name(self):
        index = ['name0', 'toy1', 'born2']
        index_gt = [0, 4, 8]
        assert IndexList(index, self.data.columns).list == index_gt

    def test_idx_by_none(self):
        index = [slice(None)]
        index_gt = list(range(len(self.data.columns)))
        assert IndexList(index, self.data.columns).list == index_gt

    def test_excel2idx(self):
        alphabets = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        excels = alphabets + [cur + ch for cur in 'ABC' for ch in alphabets]
        index_gt = list(range(len(alphabets) * 4))
        idx = [IndexList.excel2idx(elem) for elem in excels]
        assert idx == index_gt
        
