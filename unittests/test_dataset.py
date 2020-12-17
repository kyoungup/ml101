import unittest
from pathlib import Path
import pandas as pd
from sklearn.datasets import make_classification
from collections import Counter
from ml101.dataset import Sampler
import ml101.utils as utils

CWD = Path(__file__).parent
TEMP = CWD / 'temp'
TEMP.mkdir(exist_ok=True, parents=True)


class TestSampler(unittest.TestCase):
    def setUp(self) -> None:
        self.X, self.y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
                                    n_informative=3, n_redundant=1, flip_y=0,
                                    n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
        self.data_count = Counter(self.y)   #{1:900, 0:100}
        self.X = pd.DataFrame(self.X)
        self.y = pd.Series(self.y)
        self.sampler = Sampler(self.X, self.y, random_state=42)
        self.oversample = Counter({0: 900, 1: 900})
        self.oversample_numbers = Counter({0: 200, 1: 900})
        self.undersample = Counter({0: 100, 1: 100})
        self.undersample_numbers = Counter({0: 100, 1: 500})
        self.smote = Counter({0: 900, 1: 900})
        self.custom = Counter({0: 900, 1: 881})
        self.custom_numbers = Counter({0: 200, 1: 500})
        self.resample = Counter({0: 10, 1: 90})

    def tearDown(self) -> None:
        pass

    def test_oversample(self):
        _, y = self.sampler.oversample()
        assert self.sampler.count == self.oversample
    
    def test_oversample_numbers(self):
        _, y = self.sampler.oversample(method={0:200})
        assert self.sampler.count == self.oversample_numbers

    def test_smote(self):
        _, y = self.sampler.smote()
        assert self.sampler.count == self.smote

    def test_undersample(self):
        _, y = self.sampler.undersample()
        assert self.sampler.count == self.undersample

    def test_undersample_numbers(self):
        _, y = self.sampler.undersample(method={1:500})
        assert self.sampler.count == self.undersample_numbers

    def test_custom(self):
        _, y = self.sampler.custom()
        assert self.sampler.count == self.custom

    def test_custom_numbers(self):
        _, y = self.sampler.custom(methods={'smote':{0:200}, 'under':{1:500}})
        assert self.sampler.count == self.custom_numbers

    def test_resample(self):
        _, y = self.sampler.resample(n_samples=self.y.shape[0] * 0.1)
        assert self.sampler.count == self.resample
        