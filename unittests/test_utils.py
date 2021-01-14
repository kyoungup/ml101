import unittest
from pathlib import Path
import os
import ml101.utils as utils


CWD = Path(__file__).parent
TEMP = CWD / 'temp'
TEMP.mkdir(exist_ok=True, parents=True)


class TestCopy(unittest.TestCase):
    def setUp(self) -> None:
        self.src_path = TEMP / 'test1'
        self.src_path.mkdir(exist_ok=True)
        self.src_file = self.src_path / 'file.dat'
        self.src_file.touch()

        self.dst_path = TEMP / 'test2'
        self.dst_path.mkdir(exist_ok=True)
        self.dst_file = None

        self.org_path = TEMP / 'db'
        self.org_path.mkdir(exist_ok=True)
        self.org_file = self.org_path / 'file.dat'
        self.org_file.touch()

    def tearDown(self) -> None:
        if self.dst_file and (self.dst_file.is_symlink() or self.dst_file.exists()):
            self.dst_file.unlink()
        if self.dst_path.exists(): self.dst_path.rmdir()

        if self.src_file.exists(): self.src_file.unlink()
        if self.src_path.exists(): self.src_path.rmdir()
                
        if self.org_file.exists(): self.org_file.unlink()
        if self.org_path.exists(): self.org_path.rmdir()

    def test_copy(self):
        self.dst_file = utils.copy(self.src_file, self.dst_path, symbolic=False)
        assert self.dst_file == TEMP / 'test2/file.dat'

    def test_copy_symbolic(self):
        self.dst_file = utils.copy(self.src_file, self.dst_path, symbolic=True)
        if utils.is_linux():
            assert os.readlink(self.dst_file) == os.path.relpath(self.src_file, self.dst_path)
        else:
            assert self.dst_file == TEMP / 'test2/file.dat'

    def test_copy_org_symbolic(self):
        self.src_file = utils.copy(self.org_file, self.src_path, symbolic=True)
        self.dst_file = utils.copy(self.src_file, self.dst_path, symbolic=True)
        if utils.is_linux():
            assert os.readlink(self.dst_file) == os.path.relpath(self.org_file, self.dst_path)
        else:
            assert self.dst_file == TEMP / 'test2/file.dat'