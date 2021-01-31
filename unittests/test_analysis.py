import unittest
from pathlib import Path

from ml101.serialize import Stream
from ml101.analysis import Correlation
from ml101.analysis import MeanComparison


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