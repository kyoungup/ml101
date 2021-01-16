import unittest
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline

from ml101.pipeline.serialize import Data, Stream
from ml101.pipeline.preprocess import Shift, SetPeriod


CWD = Path(__file__).parent
TEMP = CWD / 'temp'
TEMP.mkdir(exist_ok=True, parents=True)


class TestPreprocessForTimeSeries(unittest.TestCase):
    def setUp(self) -> None:
        filepath = CWD / 'data' / 'stream_sample.tsv'
        self.reader = Stream(filepath)
        self.data = self.reader.read(pos_header=4)
        self.gt_period = [0, 0, 0, 0, 3, 3, 3, 0, 0, 3, 3, 3]

    def tearDown(self) -> None:
        pass

    def test_Shift(self):
        workflow = Pipeline([
            ('shift', Shift(move=-3, columns=self.data.columns))
        ])
        new_data = workflow.fit_transform(self.data.dataframe)
        assert new_data.data.shape == (self.data.shape[0] - 3, self.data.shape[1] * 2)

    def test_SetPeriod(self):
        col_new = 'time'
        col_Y = -1
        gt_data = pd.DataFrame(self.gt_period, columns=[col_new])
        workflow = Pipeline([
            ('period', SetPeriod(columns=[self.data.columns[col_Y]], outs=[col_new], append=False))
        ])
        new_data = workflow.fit_transform(self.data.dataframe)
        assert new_data[col_new].equals(gt_data[col_new])
