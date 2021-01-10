from pathlib import Path
import pandas as pd
import numpy as np
import shutil
from ml101.serialize import Stream
import ml101.utils as utils


class DataFile:
    def __init__(self, src_file, header=0) -> None:
        self.src_file = Path(src_file)
        self.reader = Stream.open(self.src_file)
        self.header = header
        self.dst_files_ = list()

    def divide_by_period(self, dst_path, date_col:str, start, end, freq='Q'):
        periods = pd.period_range(start=start, end=end, freq=freq)

        # TODO: update after time-series support
        # df = pd.read_csv(self.src_file, parse_dates=[date_col], index_col=date_col)
        df = self.reader.read(**{'header': self.header}).dataframe
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)

        for period in periods:
            chunk_file = self.src_file.stem + '_' + str(period).lower() + self.src_file.suffix
            self.dst_files_.append(chunk_file)
            # TODO: use Stream instances!
            df[str(period)].to_csv(dst_path / chunk_file)

        return self

    def divide_by_cols(self, dst_path, cols:list):
        # df = pd.read_csv(self.src_file, parse_dates=[date_col], index_col=date_col)
        df = self.reader.read(**{'header': self.header}).dataframe
        df1 = df[cols]
        df2 = df[~cols]
        file1 = utils.insert2filename(self.src_file, suffix='_inc')
        file2 = utils.insert2filename(self.src_file, suffix='_exc')
        # TODO: use Stream instances!
        df1.to_csv(file1)
        df2.to_csv(file2)

    @property
    def recent_files(self):
        return self.dst_files_

    def rename(self, colnames:dict):
        df = self.reader.read(**{'header': self.header}).dataframe
        df.rename(colnames, inplace=True)
        # construct new header
        header = ','.join(df.columns) + '\n'
        # update header in csv/tsv file
        with self.src_file.open(mode='r+') as f1, self.src_file.open(mode='r+') as f2:
            f1.readline() # to move the file pointer by header
            f2.write(header)
            shutil.copyfileobj(f1, f2)  # copy data rows
            