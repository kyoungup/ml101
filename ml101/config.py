import ml101.matplot_backend
from pathlib import Path
import shutil
from datetime import datetime
import ml101.utils as utils
import os


class Project:
    DEFAULT_NAME = 'proj'

    def __init__(self, name=DEFAULT_NAME, cwd=None):
        self.project = name if name else self.DEFAULT_NAME
        self.backup_folder = None
        self.init(cwd)

    def init(self, cwd=None):
        self.cwd = Path(cwd) / self.project if cwd else Path(__file__).parent.parent / self.project
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
            shutil.copy(previous_file, self.cwd)
        else:
            (self.cwd / 'MEMOME.md').touch()

    def __import_previous_data(self, hardcopy):
        if self.backup_folder is None:
            return

        backup_env = Project(cwd=self.backup_folder)
        raw_files = utils.listdir(backup_env.loc_data_raw)
        pure_files = utils.listdir(backup_env.loc_data_pure)
        if hardcopy:
            for file in raw_files:
                shutil.copy(file, self.loc_data_raw)
            for file in pure_files:
                shutil.copy(file, self.loc_data_pure)
        else:
            if utils.is_linux() is False:
                return
            for file in raw_files:
                (self.loc_data_raw / file.name).symlink_to(file)
            for file in pure_files:
                (self.loc_data_pure / file.name).symlink_to(file)

    def __backup(self):
        current_time = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
        self.backup_folder = self.cwd.with_name(self.cwd.name + '-' + current_time)
        self.cwd.rename(self.backup_folder)

    def new(self, name=DEFAULT_NAME, cwd=None, overwrite=False, hardcopy:bool=False):
        if self.cwd and self.cwd.exists() and overwrite is False:
            self.__backup()

        self.init(cwd)
        self.__mkdir()
        self.__prepare_memo_file()
        self.__import_previous_data(hardcopy)
        return self

    def import_rawdata(self, filepath, symbolic=True):
        self.file_raw = self.loc_data_raw / Path(filepath).name
        shutil.copy(filepath, self.file_raw)
        return self.file_raw

ENV = Project()