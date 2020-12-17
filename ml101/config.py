import ml101.matplot_backend
from pathlib import Path
import shutil
from datetime import datetime


class Project:
    DEFAULT_NAME = 'proj'
    def __init__(self, name=DEFAULT_NAME, cwd=None):
        self.project = name
        self.init(cwd)

    def __mkdir(self):
        self.loc_data_raw.mkdir(exist_ok=True, parents=True)
        self.loc_data_pure.mkdir(exist_ok=True, parents=True)
        self.loc_data_train.mkdir(exist_ok=True, parents=True)
        self.loc_data_valid.mkdir(exist_ok=True, parents=True)
        self.loc_data_test.mkdir(exist_ok=True, parents=True)

        self.loc_graphs.mkdir(exist_ok=True, parents=True)
        self.loc_graphs_data.mkdir(exist_ok=True, parents=True)
        self.loc_graphs_train.mkdir(exist_ok=True, parents=True)
        self.loc_graphs_valid.mkdir(exist_ok=True, parents=True)
        self.loc_graphs_test.mkdir(exist_ok=True, parents=True)

        self.loc_models.mkdir(exist_ok=True, parents=True)
        self.loc_results.mkdir(exist_ok=True, parents=True)

    def init(self, cwd=None):
        self.cwd = Path(cwd) / self.project if cwd else Path(__file__).parent.parent / self.project
        self.loc_data = self.cwd / 'data'
        self.loc_data_raw = self.loc_data / 'raw'
        self.loc_data_pure = self.loc_data / 'pure'
        self.loc_data_train = self.loc_data / 'train'
        self.loc_data_valid = self.loc_data / 'valid'
        self.loc_data_test = self.loc_data / 'test'

        self.loc_graphs = self.cwd / 'graphs'
        self.loc_graphs_data = self.loc_graphs / 'data'
        self.loc_graphs_train =  self.loc_graphs / 'train'
        self.loc_graphs_valid =  self.loc_graphs / 'valid'
        self.loc_graphs_test = self.loc_graphs / 'test'

        self.loc_models = self.cwd / 'models'
        self.loc_results = self.cwd / 'results'

        return self

    def new(self, name=DEFAULT_NAME, cwd=None, overwrite=False):
        if self.cwd and self.cwd.exists() and overwrite is False:
            self.__backup()

        self.init(cwd)
        self.__mkdir()
        return self

    def __backup(self):
        current_time = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
        self.cwd.rename(self.cwd.with_name(self.cwd.name + '-' + current_time))

    def import_rawdata(self, filepath):
        self.file_raw = self.loc_data_raw / Path(filepath).name
        shutil.copy(filepath, self.file_raw)
        return self.file_raw

ENV = Project()