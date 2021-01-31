import unittest
from pathlib import Path
from ml101.serialize import Stream
from ml101.cluster import KMeans


CWD = Path(__file__).parent
TEMP = CWD / 'temp'
TEMP.mkdir(exist_ok=True, parents=True)


class TestKMeans(unittest.TestCase):
    def setUp(self) -> None:
        self.stream = Stream(CWD / 'data' / 'iris.csv')
        self.data = self.stream.read()
        self.savepath = None

    def tearDown(self):
        if self.savepath and self.savepath.exists():
            self.savepath.unlink()

    def test_predict(self):
        self.alg = KMeans(self.data, cols=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], n_clusters=3)
        self.alg.cluster(n_clusters=4).predict()
        self.savepath = self.alg.show(x='sepal_length', y='petal_length', mark=True, name='Iris').save(TEMP)
        assert self.alg.centers.shape[0] == 4