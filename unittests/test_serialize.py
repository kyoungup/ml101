import unittest
from ml101.serialize import Stream
from pathlib import Path

CWD = Path(__file__).parent
TEMP = CWD / 'temp'
TEMP.mkdir(exist_ok=True, parents=True)


class TestExcel(unittest.TestCase):
    def setUp(self) -> None:
        filepath = CWD / 'data' / 'stream_sample.xlsx'
        self.reader = Stream(filepath)

        self.tempfile = TEMP / 'sample_temp.xlsx'
        self.writer = Stream(self.tempfile)

    def tearDown(self) -> None:
        if self.tempfile.exists():
            self.tempfile.unlink()

    def test_read(self):
        data = self.reader.read(sheet_name='DC', pos_header=4)
        assert data.shape == (16 - 4, ord('M') - ord('A') + 1)

    def test_write(self):
        data_gt = self.reader.read(sheet_name='DC', pos_header=4)
        self.writer.write(data_gt)

        data = self.writer.read()
        assert data.shape == data_gt.shape

    def test_write_append(self):
        data_gt = self.reader.read(sheet_name='DC', pos_header=4)
        self.writer.write(data_gt)

        data = self.writer.read()
        assert data.shape == data_gt.shape

        self.writer.write(data_gt, append=True, sheet_name='default2')
        data = self.writer.read(sheet_name='default2')
        assert data.shape == data_gt.shape


class TestCSV(unittest.TestCase):
    def setUp(self) -> None:
        filepath = CWD / 'data' / 'stream_sample.csv'
        self.reader = Stream(filepath)

        self.tempfile = TEMP / 'sample_temp.csv'
        self.writer = Stream(self.tempfile)

    def tearDown(self) -> None:
        if Path(self.tempfile).exists():
            self.tempfile.unlink()

    def test_read(self):
        data = self.reader.read(pos_header=4)
        assert data.shape == (16 - 4, 13)

    def test_write(self):
        data_gt = self.reader.read(pos_header=4)
        self.writer.write(data_gt, include_header=True)
        data = self.writer.read()
        assert data.shape == data_gt.shape


class TestTSV(unittest.TestCase):
    def setUp(self) -> None:
        filepath = CWD / 'data' / 'stream_sample.tsv'
        self.reader = Stream(filepath)

        self.tempfile = TEMP / 'sample_temp.tsv'
        self.writer = Stream(self.tempfile)

    def tearDown(self) -> None:
        if Path(self.tempfile).exists():
            self.tempfile.unlink()

    def test_read(self):
        data = self.reader.read(pos_header=4)
        assert data.shape == (16 - 4, 13)

    def test_write(self):
        data_gt = self.reader.read(pos_header=4)
        self.writer.write(data_gt, include_header=True)
        data = self.writer.read()
        assert data.shape == data_gt.shape


class TestPickle(unittest.TestCase):
    def setUp(self) -> None:
        filepath = CWD / 'data' / 'stream_sample.pkl'
        self.reader = Stream(filepath)

        self.tempfile = CWD / 'data' / 'sample_temp.pkl'
        self.writer = Stream(self.tempfile)

    def tearDown(self) -> None:
        if Path(self.tempfile).exists():
            self.tempfile.unlink()

    def test_read(self):
        data = self.reader.read()
        assert data.shape == (16 - 4, 13)

    def test_write(self):
        data_gt = self.reader.read()
        self.writer.write(data_gt)
        data = self.writer.read()
        assert data.equals(data_gt)
