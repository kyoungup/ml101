import ml101.matplot_backend
from pathlib import Path
import shutil
from datetime import datetime
import ml101.utils as utils
import os


class Project:
    DEFAULT_NAME = 'proj'
    DEFAULT_DATABANK = 'databank'

    def __init__(self, name=None, cwd=None, databank:Path=None):
        self.name = name if name else self.DEFAULT_NAME
        self.backup_folder = None
        self.databank = None
        self.init(cwd, databank)

    def init(self, cwd:Path=None, databank:Path=None):
        self.cwd = Path(cwd) / self.name if cwd else Path(__file__).parent.parent / self.name
        self.databank = Path(databank) if databank else self.cwd.parent / self.DEFAULT_DATABANK

        self.loc_data = self.cwd / 'data'
        self.loc_data_raw = self.loc_data / 'raw'
        self.loc_data_pure = self.loc_data / 'pure'
        self.loc_data_train = self.loc_data / 'train'
        self.loc_data_valid = self.loc_data / 'valid'
        self.loc_data_test = self.loc_data / 'test'

        self.loc_info = self.cwd / 'info'
        self.loc_info_raw = self.loc_info / 'raw'
        self.loc_info_pure = self.loc_info / 'pure'
        self.loc_info_train =  self.loc_info / 'train'
        self.loc_info_valid =  self.loc_info / 'valid'
        self.loc_info_test = self.loc_info / 'test'

        self.loc_models = self.cwd / 'models'
        self.loc_results = self.cwd / 'results'

        return self

    def __mkdir(self):
        self.loc_data_raw.mkdir(exist_ok=True, parents=True)
        self.loc_data_pure.mkdir(exist_ok=True, parents=True)
        self.loc_data_train.mkdir(exist_ok=True, parents=True)
        self.loc_data_valid.mkdir(exist_ok=True, parents=True)
        self.loc_data_test.mkdir(exist_ok=True, parents=True)

        self.loc_info.mkdir(exist_ok=True, parents=True)
        self.loc_info_raw.mkdir(exist_ok=True, parents=True)
        self.loc_info_pure.mkdir(exist_ok=True, parents=True)
        self.loc_info_train.mkdir(exist_ok=True, parents=True)
        self.loc_info_valid.mkdir(exist_ok=True, parents=True)
        self.loc_info_test.mkdir(exist_ok=True, parents=True)

        self.loc_models.mkdir(exist_ok=True, parents=True)
        self.loc_results.mkdir(exist_ok=True, parents=True)

    def __prepare_memo_file(self):
        if self.backup_folder:
            previous_file = self.backup_folder / 'MEMOME.md'
            utils.copy(previous_file, self.cwd)
        else:
            (self.cwd / 'MEMOME.md').touch()

    def __import_previous_data(self):
        if self.backup_folder is None:
            return

        backup_env = Project(name=self.backup_folder.name, cwd=self.backup_folder.parent)
        raw_files = utils.listdir(backup_env.loc_data_raw)
        pure_files = utils.listdir(backup_env.loc_data_pure)

        for file in raw_files:
            utils.copy(file, self.loc_data_raw)
        for file in pure_files:
            utils.copy(file, self.loc_data_pure)

    def __backup(self):
        current_time = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
        self.backup_folder = self.cwd.with_name(self.cwd.name + '-' + current_time)
        self.cwd.rename(self.backup_folder)

    def new(self, name=None, cwd=None, overwrite=False):
        if self.cwd and self.cwd.exists() and overwrite is False:
            self.__backup()

        if name: self.name = name
        self.init(cwd) if cwd else self.init(self.cwd.parent)
        self.__mkdir()
        self.__prepare_memo_file()
        self.__import_previous_data()
        return self

    def import_rawdata(self, srcfile:Path, symbolic=True):
        self.file_raw = self.loc_data_raw / Path(srcfile).name
        utils.copy(srcfile, self.file_raw, symbolic=True)
        return self.file_raw

ENV = Project()