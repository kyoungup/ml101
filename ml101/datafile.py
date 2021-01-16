from pathlib import Path
import pandas as pd
import numpy as np
import shutil
from ml101.serialize import Stream
from ml101.data import Data
import ml101.utils as utils


class DataFile:
    def __init__(self, src_file, pos_header=0) -> None:
        self.src_file = Path(src_file)
        self.reader = Stream(self.src_file)
        self.pos_header = pos_header
        self.dst_files_ = list()

    def __check_path(self, path):
        path = Path(path) if path else self.src_file.parent
        return path

    def divide_by_period(self, date_col:str, start:str=None, end:str=None, freq='Q', dst_path:Path=None):
        dst_path = self.__check_path(dst_path)
        data = self.reader.read(pos_header=self.pos_header)
        data.time_series_on(date_col)

        df = data.dataframe
        if start is None: start = data.dataframe.index.min()
        if end is None: end = data.dataframe.index.max()
        periods = pd.period_range(start=start, end=end, freq=freq)
        for period in periods:
            chunk_file = utils.insert2filename(self.src_file.name, suffix='_' + str(period).lower())
            self.dst_files_.append(chunk_file)
            Stream(dst_path / chunk_file).write(df[str(period)], index=True)
        return self

    def divide_by_cols(self, cols:list, dst_path:Path=None):
        dst_path = self.__check_path(dst_path)
        df = self.reader.read(pos_header=self.pos_header).dataframe
        df1 = df[cols]
        df2 = df[~cols]
        file1 = dst_path / utils.insert2filename(self.src_file.name, suffix='_inc')
        file2 = dst_path / utils.insert2filename(self.src_file.name, suffix='_exc')
        Stream(file1).write(Data(df1))
        Stream(file2).write(Data(df2))
        return self

    @property
    def recent_files(self):
        return self.dst_files_

    def rename(self, colnames:dict):
        df = self.reader.read(pos_header=self.pos_header).dataframe
        df.rename(colnames, inplace=True)
        # construct new header
        header = ','.join(df.columns) + '\n'
        # update header in csv/tsv file
        with self.src_file.open(mode='r+') as f1, self.src_file.open(mode='r+') as f2:
            f1.readline() # to move the file pointer by header
            f2.write(header)
            shutil.copyfileobj(f1, f2)  # copy data rows
            
