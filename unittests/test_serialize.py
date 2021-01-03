import unittest
from ml101.serialize import Stream
from pathlib import Path

CWD = Path(__file__).parent
TEMP = CWD / 'temp'
TEMP.mkdir(exist_ok=True, parents=True)


class TestExcel(unittest.TestCase):
    def setUp(self) -> None:
        filepath = CWD / 'data' / 'stream_sample.xlsx'
        self.readstream = Stream.open(filepath)
        self.readoption = {'sheet_name': 'DC', 'header': 4}

        self.tempfile = TEMP / 'sample_temp.xlsx'
        self.writestream = Stream.open(self.tempfile)

    def tearDown(self) -> None:
        if self.tempfile.exists():
            self.tempfile.unlink()

    def test_read(self):
        data = self.readstream.read(**self.readoption)
        assert data.shape == (16 - 4, ord('M') - ord('A') + 1)

    def test_write(self):
        data_gt = self.readstream.read(**self.readoption)
        self.writestream.write(data_gt)

        data = self.writestream.read()
        assert data.shape == data_gt.shape

    def test_write_append(self):
        data_gt = self.readstream.read(**self.readoption)
        self.writestream.write(data_gt)

        data = self.writestream.read(**self.writestream.default)
        assert data.shape == data_gt.shape

        self.writestream.write(data_gt, append=True, **{'sheet_name': 'default2'})
        data = self.writestream.read(**{'sheet_name': 'default2'})
        assert data.shape == data_gt.shape

    def test_default(self):
        assert self.readstream.default is not None


class TestCSV(unittest.TestCase):
    def setUp(self) -> None:
        filepath = CWD / 'data' / 'stream_sample.csv'
        self.readstream = Stream.open(filepath)
        self.readoption = {'header': 4}

        self.tempfile = TEMP / 'sample_temp.csv'
        self.writestream = Stream.open(self.tempfile)

    def tearDown(self) -> None:
        if Path(self.tempfile).exists():
            self.tempfile.unlink()

    def test_read(self):
        data = self.readstream.read(**self.readoption)
        assert data.shape == (16 - 4, 13)

    def test_write(self):
        data_gt = self.readstream.read(**self.readoption)
        self.writestream.write(data_gt, **{'header': True})
        data = self.writestream.read(**{'header': 0})
        assert data.shape == data_gt.shape


class TestTSV(unittest.TestCase):
    def setUp(self) -> None:
        filepath = CWD / 'data' / 'stream_sample.tsv'
        self.readstream = Stream.open(filepath)
        self.readoption = {'header': 4}

        self.tempfile = TEMP / 'sample_temp.tsv'
        self.writestream = Stream.open(self.tempfile)

    def tearDown(self) -> None:
        if Path(self.tempfile).exists():
            self.tempfile.unlink()

    def test_read(self):
        data = self.readstream.read(**self.readoption)
        assert data.shape == (16 - 4, 13)

    def test_write(self):
        data_gt = self.readstream.read(**self.readoption)
        self.writestream.write(data_gt, **{'header': True})
        data = self.writestream.read(**{'header': 0})
        assert data.shape == data_gt.shape


class TestPickle(unittest.TestCase):
    def setUp(self) -> None:
        filepath = CWD / 'data' / 'stream_sample.pkl'
        self.readstream = Stream.open(filepath)

        self.tempfile = CWD / 'data' / 'sample_temp.pkl'
        self.writestream = Stream.open(self.tempfile)

    def tearDown(self) -> None:
        if Path(self.tempfile).exists():
            self.tempfile.unlink()

    def test_read(self):
        data = self.readstream.read()
        assert data.shape == (16 - 4, 13)

    def test_write(self):
        data_gt = self.readstream.read()
        self.writestream.write(data_gt)
        data = self.writestream.read()
        assert data.equals(data_gt)
