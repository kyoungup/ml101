import unittest
import shutil
from pathlib import Path
from ml101.config import Project


CWD = Path(__file__).parent
TEMP = CWD / 'temp'
TEMP.mkdir(exist_ok=True, parents=True)


class TestProejct(unittest.TestCase):
    def setUp(self) -> None:
        self.ENV = Project(name='test', cwd=TEMP)

    def tearDown(self) -> None:
        if self.ENV.cwd.exists():
            shutil.rmtree(self.ENV.cwd)
        if self.ENV.loc_databank.exists():
            shutil.rmtree(self.ENV.loc_databank)
    
    def test_new(self):
        ENV = self.ENV.new()
        assert ENV.cwd.exists()

    def test_init(self):
        assert self.ENV.loc_data_raw == self.ENV.loc_data / 'raw'
        assert self.ENV.loc_data_pure == self.ENV.loc_data / 'pure'
        assert self.ENV.loc_data_train == self.ENV.loc_data / 'train'
        assert self.ENV.loc_data_valid == self.ENV.loc_data / 'valid'
        assert self.ENV.loc_data_test == self.ENV.loc_data / 'test'

        assert self.ENV.loc_info_raw == self.ENV.loc_info / 'raw'
        assert self.ENV.loc_info_pure == self.ENV.loc_info / 'pure'
        assert self.ENV.loc_info_train == self.ENV.loc_info / 'train'
        assert self.ENV.loc_info_valid == self.ENV.loc_info / 'valid'
        assert self.ENV.loc_info_test == self.ENV.loc_info / 'test'
        
        assert self.ENV.loc_models == self.ENV.cwd / 'models'
        assert self.ENV.loc_results == self.ENV.cwd / 'results'