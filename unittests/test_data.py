import unittest
from ml101.serialize import Stream
from ml101.data import Data, IndexList
from pathlib import Path

CWD = Path(__file__).parent
TEMP = CWD / 'temp'
TEMP.mkdir(exist_ok=True, parents=True)


class TestData(unittest.TestCase):
    def setUp(self) -> None:
        filepath = CWD / 'data' / 'stream_sample.tsv'
        self.stream = Stream.open(filepath)
        self.data = self.stream.read(**{'header': 4})

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

    def test_na_ratio(self):
        assert (self.data.na_ratio('row') > 0).sum() == 6


class TestIndexList(unittest.TestCase):
    def setUp(self) -> None:
        self.maxlen = 1000

        filepath = CWD / 'data' / 'stream_sample.tsv'
        self.stream = Stream.open(filepath)
        self.data = self.stream.read(**{'header': 4})

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
        