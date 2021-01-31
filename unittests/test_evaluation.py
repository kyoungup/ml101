import unittest
from pathlib import Path
import numpy as np

from ml101.evaluation import Scores
from ml101.evaluation import CScores, CAggr
from ml101.evaluation import RScores, RAggr
from ml101.evaluation import SimpleScores
from ml101.evaluation import ConfusionMatrixGraph
import ml101.utils as utils


CWD = Path(__file__).parent
TEMP = CWD / 'temp'
TEMP.mkdir(exist_ok=True, parents=True)


class TestCScores(unittest.TestCase):
    def setUp(self) -> None:
        self.true_y1 = [0, 1, 2, 2, 2]
        self.pred_y1 = [0, 0, 2, 2, 1]
        labels = {0: 'class 0', 1: 'class 1', 2: 'class 2'}
        self.scores1 = CScores(TEMP, prefix='test1_', idx2label=labels)
        self.report1 = {'accuracy': 0.6,
                        'class 0': {'f1-score': 0.6666666666666666,'precision': 0.5,'recall': 1.0,'support': 1},
                        'class 1': {'f1-score': 0.0, 'precision': 0.0, 'recall': 0.0, 'support': 1},
                        'class 2': {'f1-score': 0.8, 'precision': 1.0, 'recall': 0.6666666666666666, 'support': 3},
                        'macro avg': {'f1-score': 0.48888888888888893, 'precision': 0.5, 'recall': 0.5555555555555555, 'support': 5},
                        'weighted avg': {'f1-score': 0.6133333333333334, 'precision': 0.7, 'recall': 0.6, 'support': 5}}
        self.report_classes1 = {'class 0': {'f1-score': 0.6666666666666666,'precision': 0.5,'recall': 1.0,'support': 1},
                        'class 1': {'f1-score': 0.0, 'precision': 0.0, 'recall': 0.0, 'support': 1},
                        'class 2': {'f1-score': 0.8, 'precision': 1.0, 'recall': 0.6666666666666666, 'support': 3}}
        self.cm1 = np.array([[1, 0, 0],[1, 0, 0],[0, 1, 2]])

        self.true_y2 = ["cat", "ant", "cat", "cat", "ant", "bird"]
        self.pred_y2 = ["ant", "ant", "cat", "cat", "ant", "cat"]
        self.scores2 = CScores(TEMP, prefix='test2_')
        self.report2 = {'accuracy': 0.6666666666666666,
                        'ant': {'f1-score': 0.8, 'precision': 0.6666666666666666, 'recall': 1.0, 'support': 2},
                        'bird': {'f1-score': 0.0, 'precision': 0.0, 'recall': 0.0, 'support': 1},
                        'cat': {'f1-score': 0.6666666666666666, 'precision': 0.6666666666666666, 'recall': 0.6666666666666666, 'support': 3},
                        'macro avg': {'f1-score': 0.48888888888888893, 'precision': 0.4444444444444444, 'recall': 0.5555555555555555, 'support': 6},
                        'weighted avg': {'f1-score': 0.6, 'precision': 0.5555555555555555, 'recall': 0.6666666666666666, 'support': 6}}
        self.cm2 = np.array([[2, 0, 0], [0, 0, 1], [1, 0, 2]])

    def tearDown(self) -> None:
        if (TEMP / ('test1_' + CScores.FILENAME)).exists():
            (TEMP / ('test1_' + CScores.FILENAME)).unlink()
        if (TEMP / ('test2_' + CScores.FILENAME)).exists():
            (TEMP / ('test2_' + CScores.FILENAME)).unlink()

    def test_confusion_matrix(self):
        self.scores1.add(self.true_y1, self.pred_y1, Scores.TEST)
        assert np.allclose(self.cm1, self.scores1.confusion_matrix(Scores.TEST), atol=0.0001)

    def test_report_class(self):
        self.scores1.add(self.true_y1, self.pred_y1, Scores.TEST)
        assert utils.round_container(self.report_classes1) == utils.round_container(self.scores1.report_classes(Scores.TEST))

    def test_add_save(self):
        self.scores1.add(self.true_y1, self.pred_y1, Scores.TRAIN)
        self.scores1.add(self.true_y1, self.pred_y1, Scores.TEST)
        assert utils.round_container(self.report1) == utils.round_container(self.scores1.scores(Scores.TRAIN))
        assert utils.round_container(self.report1) == utils.round_container(self.scores1.scores(Scores.TEST))

        self.scores1.save()
        assert (TEMP / ('test1_' + CScores.FILENAME)).exists()

    def test_add_by_str_save(self):
        self.scores2.add(self.true_y2, self.pred_y2, Scores.TRAIN)
        self.scores2.add(self.true_y2, self.pred_y2, Scores.TEST)
        assert utils.round_container(self.report2) == utils.round_container(self.scores2.scores(Scores.TRAIN))
        assert utils.round_container(self.report2) == utils.round_container(self.scores2.scores(Scores.TEST))

        self.scores2.save()
        assert (TEMP / ('test2_' + CScores.FILENAME)).exists()

    def test_load(self):
        self.scores1.add(self.true_y1, self.pred_y1, Scores.TRAIN)
        self.scores1.add(self.true_y1, self.pred_y1, Scores.TEST)
        self.scores1.save()

        scores = CScores(TEMP, prefix='test1_')
        scores.load()
        assert utils.round_container(self.report1) == utils.round_container(scores.scores(Scores.TRAIN))
        assert utils.round_container(self.report1) == utils.round_container(scores.scores(Scores.TEST))
        assert np.allclose(self.cm1, scores.confusion_matrix(Scores.TEST), atol=0.0001)


class TestCAggScores(unittest.TestCase):
    def setUp(self) -> None:
        self.true_y1 = [0, 1, 2, 2, 2]
        self.pred_y1 = [0, 0, 2, 2, 1]
        labels = {0: 'class 0', 1: 'class 1', 2: 'class 2'}
        self.scores1 = CScores(TEMP, prefix='test1_', idx2label=labels)
        self.scores1.add(self.true_y1, self.pred_y1, Scores.TEST)

        self.scores2 = CScores(TEMP, prefix='test2_', idx2label=labels)
        self.scores2.add(self.true_y1, self.pred_y1, Scores.TEST)

        self.agg = CAggr(TEMP, 'test_')
        self.agg_report_test = {'accuracy': {'f1-score': 0.6, 'precision': 0.6, 'recall': 0.6, 'support': 0.6},
                                'class 0': {'f1-score': 0.6666666666666666, 'precision': 0.5, 'recall': 1.0, 'support': 1.0},
                                'class 1': {'f1-score': 0.0, 'precision': 0.0, 'recall': 0.0, 'support': 1.0},
                                'class 2': {'f1-score': 0.8, 'precision': 1.0, 'recall': 0.6666666666666666, 'support': 3.0},
                                'macro avg': {'f1-score': 0.48888888888888893, 'precision': 0.5, 'recall': 0.5555555555555555, 'support': 5.0},
                                'weighted avg': {'f1-score': 0.6133333333333334, 'precision': 0.7, 'recall': 0.6, 'support': 5.0}}
        self.agg_cm_test = np.array([[1, 0, 0],[1, 0, 0],[0, 1, 2]])

    def tearDown(self) -> None:
        if (TEMP / ('test1_' + CScores.FILENAME)).exists():
            (TEMP / ('test1_' + CScores.FILENAME)).unlink()
        if (TEMP / ('test2_' + CScores.FILENAME)).exists():
            (TEMP / ('test2_' + CScores.FILENAME)).unlink()
        if (TEMP / ('test_' + CAggr.FILENAME)).exists():
            (TEMP / ('test_' + CAggr.FILENAME)).unlink()

    def test_add_save(self):
        self.agg.add(self.scores1, self.scores2)
        assert utils.round_container(self.agg_report_test) == utils.round_container(self.agg.metrics[Scores.TEST])
        assert np.allclose(self.agg_cm_test, self.agg.confusion_matrices[Scores.TEST], atol=0.1)

        self.agg.save()
        assert (TEMP / ('test1_' + CScores.FILENAME)).exists()
        assert (TEMP / ('test2_' + CScores.FILENAME)).exists()
        assert (TEMP / ('test_' + CAggr.FILENAME)).exists()

    def test_load(self):
        self.agg.add(self.scores1, self.scores2)
        self.agg.save()

        agg = CAggr(TEMP, 'test_')
        agg.load()
        assert utils.round_container(self.agg_report_test) == utils.round_container(agg.metrics[Scores.TEST])
        assert np.allclose(self.agg_cm_test, agg.confusion_matrices[Scores.TEST], atol=0.1)


class TestRScores(unittest.TestCase):
    def setUp(self) -> None:
        self.true_y1 = [3, -0.5, 2, 7]
        self.pred_y1 = [2.5, 0.0, 2, 8]
        self.scores1 = RScores(TEMP, prefix='test_')
        self.report1 = {RScores.MSE: 0.61237243569579450,
                        RScores.RMSE: 0.375,
                        RScores.MAE: 0.5,
                        RScores.MDAE: 0.5,
                        RScores.R2: 0.9486081370449679,
                        RScores.EV: 0.9571734475374732,
                        RScores.MSLE: -np.inf}

        # self.y_true2 = [3, 5, 2.5, 7]
        # self.y_pred2 = [2.5, 5, 4, 8]
        # self.report2 = dict(mean_squared_log_error = 0.03973012298459379)

    def tearDown(self) -> None:
        if (TEMP / ('test_' + RScores.FILENAME)).exists():
            (TEMP / ('test_' + RScores.FILENAME)).unlink()

    def test_mean_squared_error(self):
        self.scores1.add(self.true_y1, self.pred_y1, Scores.TEST)
        assert round(self.report1['mean_squared_error'], ndigits=4) == round(self.scores1.mean_squared_error(Scores.TEST), ndigits=4)

    def test_add_save(self):
        self.scores1.add(self.true_y1, self.pred_y1, Scores.TRAIN)
        self.scores1.add(self.true_y1, self.pred_y1, Scores.TEST)
        assert utils.round_container(self.report1) == utils.round_container(self.scores1.scores(Scores.TRAIN))
        assert utils.round_container(self.report1) == utils.round_container(self.scores1.scores(Scores.TEST))

        self.scores1.save()
        assert (TEMP / ('test_' + RScores.FILENAME)).exists()

    def test_load(self):
        self.scores1.add(self.true_y1, self.pred_y1, Scores.TRAIN)
        self.scores1.save()

        scores = RScores(TEMP, prefix='test_')
        scores.load()

        assert utils.round_container(self.report1) == utils.round_container(scores.scores(Scores.TRAIN))


class TestRAggScores(unittest.TestCase):
    def setUp(self) -> None:
        self.true_y1 = [3, -0.5, 2, 7]
        self.pred_y1 = [2.5, 0.0, 2, 8]
        self.scores1 = RScores(TEMP, prefix='test1_')
        self.scores1.add(self.true_y1, self.pred_y1, Scores.TEST)
        self.scores2 = RScores(TEMP, prefix='test2_')
        self.scores2.add(self.true_y1, self.pred_y1, Scores.TEST)

        self.agg = RAggr(TEMP, 'test_')
        self.agg_report_test = {RScores.MSE: 0.61237243569579450,
                                RScores.RMSE: 0.375,
                                RScores.MAE: 0.5,
                                RScores.MDAE: 0.5,
                                RScores.R2: 0.9486081370449679,
                                RScores.EV: 0.9571734475374732,
                                RScores.MSLE: -np.inf}

    def tearDown(self) -> None:
        if (TEMP / ('test1_' + RScores.FILENAME)).exists():
            (TEMP / ('test1_' + RScores.FILENAME)).unlink()
        if (TEMP / ('test2_' + RScores.FILENAME)).exists():
            (TEMP / ('test2_' + RScores.FILENAME)).unlink()
        if (TEMP / ('test_' + RAggr.FILENAME)).exists():
            (TEMP / ('test_' + RAggr.FILENAME)).unlink()

    def test_add_save(self):
        self.agg.add(self.scores1, self.scores2)
        assert utils.round_container(self.agg_report_test) == utils.round_container(self.agg.metrics[Scores.TEST])

        self.agg.save()
        assert (TEMP / ('test1_' + RScores.FILENAME)).exists()
        assert (TEMP / ('test1_' + RScores.FILENAME)).exists()
        assert (TEMP / ('test_' + RAggr.FILENAME)).exists()

    def test_load(self):
        self.agg.add(self.scores1, self.scores2)
        self.agg.save()

        agg = RAggr(TEMP, 'test_')
        agg.load()
        assert utils.round_container(self.agg_report_test) == utils.round_container(agg.metrics[Scores.TEST])


class TestSimpleScores(unittest.TestCase):
    def setUp(self) -> None:
        self.data = dict(row1=dict(train=100.0, valid=95.0, test=90.0),
                        row2=dict(train=10.0, valid=9.5, test=9.0))
        self.data_agg = dict(mean=dict(train=55.0, valid=52.25, test=49.5),
                            std=dict(train=63.639610, valid=60.457630, test=57.275649))
        self.scores = SimpleScores(TEMP, prefix='test_')

    def tearDown(self) -> None:
        if (TEMP / 'test_scores.json').exists():
            (TEMP / 'test_scores.json').unlink()
        if (TEMP / 'test_aggregation.json').exists():
            (TEMP / 'test_aggregation.json').unlink()

    def test_set(self):
        for key, value in self.data.items():
            self.scores.set(name=key, **value)
            assert self.scores.get(key) == value

    def test_aggregate(self):
        for key, value in self.data.items():
            self.scores.set(name=key, **value)
        self.scores.mean()
        assert self.scores.scores_mean == self.data_agg['mean']
        self.scores.std()
        assert {key: round(value, 4) for key, value in self.scores.scores_std.items()} ==\
            {key: round(value, 4) for key, value in self.data_agg['std'].items()}

    def save(self):
        self.scores.save()
        assert (TEMP / 'test_scores.json').exists()
        assert (TEMP / 'test_aggregation.json').exists()


class TestConfusionMatrixGraph(unittest.TestCase):
    def setUp(self):
        self.savefile = None
        self.data = (np.random.rand(10, 10) * 100).astype(int)
        self.g = ConfusionMatrixGraph(cm=self.data, name='corr')

    def tearDown(self) -> None:
        if self.savefile and self.savefile.exists():
            self.savefile.unlink()

    def test_graph_cm(self):
        self.g.draw(suffix='_Random Correlation')
        self.savefile = self.g.save(TEMP)
        assert self.savefile.exists()