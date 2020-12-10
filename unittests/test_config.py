import unittest
import shutil
from pathlib import Path
from uids.config import Project


CWD = Path(__file__).parent


class TestProejct(unittest.TestCase):
    def setUp(self) -> None:
        self.ENV = Project(name='test')

    def tearDown(self) -> None:
        if self.ENV.cwd.exists():
            shutil.rmtree(self.ENV.cwd)
    
    def test_new(self):
        ENV = self.ENV.new(cwd=CWD / 'temp')
        assert ENV.cwd.exists()

    def test_init(self):
        self.ENV.init(cwd=CWD / 'temp')
        assert self.ENV.loc_data_raw == self.ENV.loc_data / 'raw'
        assert self.ENV.loc_data_pure == self.ENV.loc_data / 'pure'
        assert self.ENV.loc_data_train == self.ENV.loc_data / 'train'
        assert self.ENV.loc_data_valid == self.ENV.loc_data / 'valid'
        assert self.ENV.loc_data_test == self.ENV.loc_data / 'test'

        assert self.ENV.loc_graphs_data == self.ENV.loc_graphs / 'data'
        assert self.ENV.loc_graphs_train == self.ENV.loc_graphs / 'train'
        assert self.ENV.loc_graphs_valid == self.ENV.loc_graphs / 'valid'
        assert self.ENV.loc_graphs_test == self.ENV.loc_graphs / 'test'

        assert self.ENV.TEMP == self.ENV.cwd / 'results'