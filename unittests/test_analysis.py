import unittest
from pathlib import Path
import seaborn as sns
import pandas as pd
from ml101.serialize import Stream
from ml101.analysis import Correlation
from ml101.analysis import MeanComparison
from ml101.analysis import Interval


CWD = Path(__file__).parent
TEMP = CWD / 'temp'
TEMP.mkdir(exist_ok=True, parents=True)


class TestCorrelation(unittest.TestCase):
    def setUp(self) -> None:
        self.reader = Stream(CWD / 'data' / 'penguins.csv')
        self.data = self.reader.read()
        self.savepath = None
        self.corr = Correlation(self.data)

    def tearDown(self):
        if self.savepath and self.savepath.exists():
            self.savepath.unlink()

    def test_correlation_all(self):
        corr = self.corr.infer().corr
        assert corr.shape == (4, 4)

        sorted = self.corr.sort()
        assert sorted.shape == (6, )
        
        self.savepath = self.corr.show().save(TEMP)
        assert self.savepath and self.savepath.exists()

    def test_correlation_withY(self):
        corr = self.corr.infer(col='body_mass_g').corr
        assert corr.shape == (3, )

        sorted = self.corr.sort()
        assert sorted.shape == (3, )


class TestMeanComparison(unittest.TestCase):
    def setUp(self) -> None:
        self.data = Stream(CWD / 'data' / 'iris.csv').read()
        self.mc = MeanComparison(self.data)

    def tearDown(self):
        pass

    def test_derive_significant_variables(self):
        Significant_variables = self.mc.siginificant_variables(group='species')
        assert len(Significant_variables) == 4

    def test_derive_significant_variables_small_threshold(self):
        Significant_variables = self.mc.siginificant_variables(group='species', threshold=1e-40)
        assert len(Significant_variables) == 2
    
    def test_derive_significant_variables_with_selected_columns_and_small_threshold(self):
        Significant_variables = self.mc.siginificant_variables(group='species', threshold=1e-40, cols=['sepal_width', 'petal_length'])
        assert len(Significant_variables) == 1

    def test_derive_significant_variables_for_two_groups(self):
        df = self.data.dataframe
        self.mc.data.dataframe = df[df['species'] != 'setosa']
        Significant_variables = self.mc.siginificant_variables(group='species')
        assert len(Significant_variables) == 4

    def test_derive_significant_variables_for_two_groups_with_selected_columns_and_small_threshold(self):
        df = self.data.dataframe
        self.mc.data.dataframe = df[df['species'] != 'setosa']
        Significant_variables = self.mc.siginificant_variables(group='species', threshold=1e-40, cols=['sepal_width', 'petal_length'])
        assert len(Significant_variables) == 0


class TestInterval(unittest.TestCase):
    def setUp(self) -> None:
        self.savefile = None
        self.data = sns.load_dataset('iris')

    def tearDown(self) -> None:
        if self.savefile and self.savefile.exists():
            self.savefile.unlink()

    def test_draw(self):
        self.g = Interval(name='PetalLength', data=self.data, x='species', y='petal_length', join=False)
        self.g.draw(capsize=0.05, scale=0.8, errwidth=2)
        self.savefile = self.g.save(TEMP)
        assert self.savefile.exists()

    def test_draw_with_confidence_label(self):
        self.g = Interval(name='PetalLength_99', data=self.data, x='species', y='petal_length', confidence=99)
        self.g.draw()
        self.savefile = self.g.save(TEMP)
        assert self.savefile.exists()

    def test_calc_interval(self):
        gt_intervals = {'category': ['mean', 'lower', 'upper'],
                        'versicolor': [4.26, 4.126452778080923, 4.393547221919077],
                        'setosa': [1.4620000000000002, 1.412645238352349, 1.5113547616476515],
                        'virginica': [5.5520000000000005, 5.395153263133577, 5.708846736866424]}
        intervals = Interval.calc_intervals(self.data, 'species', 'petal_length')

        df_gt = pd.DataFrame(gt_intervals).set_index('category').T.sort_index()
        df_intervals = pd.DataFrame(intervals).set_index('category').T.sort_index()
        assert df_gt.round(4).equals(df_intervals.round(4))