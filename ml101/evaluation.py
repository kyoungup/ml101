from abc import ABCMeta, abstractmethod
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import explained_variance_score, mean_absolute_error
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.metrics import median_absolute_error, r2_score
import numpy as np
import itertools
import ml101.utils as utils
from ml101.graphs import Heatmap
from ml101.data import TDATA


class Scores(metaclass=ABCMeta):
    # Experiment types to evaluate
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'

    DEFAULT_NAME = 'default'
    FILENAME = 'scores.json'
    
    def __init__(self, path2file, prefix=None):
        self.path, self.filename = utils.confirm_path(path2file)
        if self.filename is None: self.filename = self.FILENAME
        self.prefix = prefix

        self.metrics = defaultdict(dict)

    def report(self, exp_set=TEST) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.scores(exp_set), orient='columns').T

    @abstractmethod
    def add(self, true_y, pred_y):
        pass

    def scores(self, exp_set=TEST):
        return self.metrics[exp_set]

    def save(self, results=dict()):
        results['metrics'] = self.metrics

        score_file = self.path / self.filename

        if self.prefix:
            score_file = utils.insert2filename(score_file, prefix=self.prefix)
            results['prefix'] = self.prefix

        serialized_results = utils.convert4json(results)
        with score_file.open(mode='w') as f:
            json.dump(serialized_results, f, indent=4)
        return score_file

    def load(self):
        score_file = self.path / self.filename

        if self.prefix:
            score_file = utils.insert2filename(score_file, prefix=self.prefix)

        with score_file.open(mode='r') as f:
            serialized_results = json.load(f)

        if 'prefix' in serialized_results:
            self.prefix = serialized_results['prefix']

        self.metrics = serialized_results['metrics']

        return serialized_results


class CScores(Scores):
    # columns in scikit-learn 'classification report'
    MICRO = 'micro avg'
    MACRO = 'macro avg'
    WEIGHTED = 'weighted avg'

    FILENAME = 'scores_clf.json'

    def __init__(self, path2file, prefix=None, idx2label:dict=None):
        super().__init__(path2file, prefix)
        self.idx2label = idx2label
        self.confusion_matrices = defaultdict(dict)

    def add(self, true_y, pred_y, exp_set=Scores.TEST):
        assert len(true_y) == len(pred_y)

        if self.idx2label is None:            
            self.idx2label = utils.build_idx2labels(true_y, pred_y) 
        
        if any(isinstance(label, str) for label in true_y) or any(isinstance(label, str) for label in pred_y):
            labels_order = list(self.idx2label.values())
        else:
            labels_order = list(self.idx2label.keys())
        show_names = list(self.idx2label.values())
        
        report = classification_report(y_true=true_y, y_pred=pred_y, output_dict=True,
                                       labels=labels_order, target_names=show_names)
        self.metrics[exp_set] = report

        self.confusion_matrices[exp_set] = confusion_matrix(y_true=true_y, y_pred=pred_y, labels=labels_order)
        return report

    def acc(self, exp_set=Scores.TEST):
        if isinstance(self.metrics[exp_set]['accuracy'], dict):
            return self.metrics[exp_set]['accuracy']['precision']
        return self.metrics[exp_set]['accuracy']

    def precision(self, exp_set=Scores.TEST, avg=MACRO, label=None):
        sel = label if label else avg
        return self.metrics[exp_set][sel]['precision']

    def recall(self, exp_set=Scores.TEST, avg=MACRO, label=None):
        sel = label if label else avg
        return self.metrics[exp_set][sel]['recall']

    def f1(self, exp_set=Scores.TEST, avg=MACRO, label=None):
        sel = label if label else avg
        return self.metrics[exp_set][sel]['f1-score']

    def confusion_matrix(self, exp_set=Scores.TEST) -> np.ndarray:
        return self.confusion_matrices[exp_set]

    def report_classes(self, exp_set=Scores.TEST):
        class_report = self.metrics[exp_set].copy()
        del class_report['accuracy']
        if self.MICRO in class_report: del class_report[self.MICRO]
        del class_report[self.MACRO], class_report[self.WEIGHTED]
        return class_report

    def save(self, results=dict()):
        results['confusion_matrix'] = self.confusion_matrices
        return super().save(results)

    def load(self):
        serialized_results = super().load()
        for name, cm in serialized_results['confusion_matrix'].items():
            self.confusion_matrices[name] = np.array(cm)
        return serialized_results


class AggScores(Scores, metaclass=ABCMeta):
    FILENAME = 'aggregation.json'

    def __init__(self, path2file, prefix=None):
        super().__init__(path2file, prefix)
        self.list = list()        
        self.metrics_stds = defaultdict(dict)

    @abstractmethod
    def _update(self):
        pass

    def add(self, *scores):
        assert all(isinstance(score, Scores) for score in scores)
        self.list.extend(scores)
        self._update()
        return self

    def std(self, exp_set=Scores.TEST):
        return self.metrics_stds[exp_set]

    def __save_each(self) -> list:
        paths = list()
        for score in self.list:
            path = score.save()
            paths.append(path)
        return paths

    def save(self, results=dict(), each=True):
        if each:
            results['paths'] = self.__save_each()

        results['metrics_std'] = self.metrics_stds
        return super().save(results)

    def _load_each(self, paths):
        for filename in paths:
            if isinstance(self, CScores):
                scores = CScores(filename)
                scores.load()
            elif isinstance(self, RScores):
                scores = RScores(filename)
                scores.load()
            else:
                raise ValueError(f'Unsupported  class: {type(self)}')
            self.list.append(scores)

    def load(self):
        serialized_results = super().load()

        if 'paths' in serialized_results:
            self._load_each(serialized_results['paths'])
            # del serialized_results['paths']
        self.metrics_stds = serialized_results['metrics_std']

        return serialized_results


class CAggr(AggScores, CScores):
    FILENAME = 'aggregation_clf.json'

    def __init__(self, path2file, prefix=None, idx2label:dict=None):
        super().__init__(path2file, prefix)
        self.idx2label = idx2label
        self.confusion_matrices_stds = defaultdict(dict)    

    def _update(self):
        for exp_set in [self.TRAIN, self.VALID, self.TEST]:
            reports = list()
            for score in self.list:
                report = pd.DataFrame.from_dict(score.scores(exp_set), orient='columns').T
                reports.append(report)
            df_reports = pd.concat(reports)
            by_row_index = df_reports.groupby(df_reports.index)
            df_means = by_row_index.mean() if len(by_row_index.groups) else pd.DataFrame()
            df_stds = by_row_index.std() if len(by_row_index.groups) else pd.DataFrame()
            self.metrics[exp_set] = df_means.to_dict(orient='index')
            self.metrics_stds[exp_set] = df_stds.to_dict(orient='index')

            cmatrices = list()
            for score in self.list:
                cm = pd.DataFrame(score.confusion_matrix(exp_set))
                cmatrices.append(cm)
            df_cms = pd.concat(cmatrices)
            by_row_index = df_cms.groupby(df_cms.index)
            df_means = by_row_index.mean() if len(by_row_index.groups) else pd.DataFrame()
            df_stds = by_row_index.std() if len(by_row_index.groups) else pd.DataFrame()
            self.confusion_matrices[exp_set] = df_means.to_numpy()
            self.confusion_matrices_stds[exp_set] = df_stds.to_numpy()

    def save(self):
        results = dict(confusion_matrix_std=self.confusion_matrices_stds)
        return super().save(results)

    def load(self):
        serialized_results = super().load()
        for name, cm in serialized_results['confusion_matrix_std'].items():
            self.confusion_matrices_stds[name] = np.array(cm)
        return serialized_results


class RScores(Scores):
    # regression losses in scikit-learn
    MAE = 'mean_absolute_error'
    MDAE = 'median_absolute_error'
    MSE = 'mean_squared_error'
    RMSE = 'root_mean_squared_error'
    MSLE = 'mean_squared_log_error'
    R2 ='R^2_score'
    EV = 'explained_variance_score'

    FILENAME = 'scores_reg.json'

    def add(self, true_y, pred_y, exp_set=Scores.TEST):
        assert len(true_y) == len(pred_y)

        # Regression metrics
        self.metrics[exp_set][RScores.MSE] = mean_squared_error(true_y, pred_y, squared=False)
        self.metrics[exp_set][RScores.RMSE] = mean_squared_error(true_y, pred_y, squared=True)
        if (np.array(true_y) > 0).all() and (np.array(pred_y) > 0).all():
            self.metrics[exp_set][RScores.MSLE] = mean_squared_log_error(true_y, pred_y)
        else:
            self.metrics[exp_set][RScores.MSLE] = -np.inf
        self.metrics[exp_set][RScores.MAE] = mean_absolute_error(true_y, pred_y)
        self.metrics[exp_set][RScores.MDAE] = median_absolute_error(true_y, pred_y)
        self.metrics[exp_set][RScores.R2] = r2_score(true_y, pred_y)
        self.metrics[exp_set][RScores.EV] = explained_variance_score(true_y, pred_y)

    def report(self, exp_set=Scores.TEST) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.scores(exp_set), orient='index')

    def mean_squared_error(self, exp_set=Scores.TEST):
        return self.metrics[exp_set][RScores.MSE]

    def root_mean_squared_error(self, exp_set=Scores.TEST):
        return self.metrics[exp_set][RScores.RMSE]

    def mean_squared_log_error(self, exp_set=Scores.TEST):
        return self.metrics[exp_set][RScores.MSLE]

    def mean_absolute_error(self, exp_set=Scores.TEST):
        return self.metrics[exp_set][RScores.MAE]

    def median_absolute_error(self, exp_set=Scores.TEST):
        return self.metrics[exp_set][RScores.MDAE]

    def r2(self, exp_set=Scores.TEST):
        return self.metrics[exp_set][RScores.R2]

    def explained_variance(self, exp_set=Scores.TEST):
        return self.metrics[exp_set][RScores.EV]


class RAggr(AggScores, RScores):
    FILENAME = 'aggregation_reg.json'

    def _update(self):
        for exp_set in [self.TRAIN, self.VALID, self.TEST]:
            reports = list()
            for score in self.list:
                report = pd.DataFrame.from_dict(score.scores(exp_set), orient='index')
                reports.append(report)
            df_reports = pd.concat(reports)
            by_row_index = df_reports.groupby(df_reports.index)
            df_means = by_row_index.mean()[0] if len(by_row_index.groups) else pd.Series()
            df_stds = by_row_index.std()[0] if len(by_row_index.groups) else pd.Series()
            self.metrics[exp_set] = df_means.to_dict()
            self.metrics_stds[exp_set] = df_stds.to_dict()


class SimpleScores:
    DEFAULT_NAME = 'default'
    SCORE_FILE = 'scores.json'
    AGGREGATE_FILE = 'aggregation.json'

    def __init__(self, path2file, prefix=None):
        self.path2file = Path(path2file).parent if Path(path2file) else Path(path2file)
        self.prefix = prefix

        self.scores = defaultdict(dict)
        self.scores_mean = None
        self.scores_std = None

    def __get_param(self, name, **kwargs):
        param = None
        if name in kwargs:
            param = kwargs[name]
            if not isinstance(param, float) and not isinstance(param, int):
                param = None
            else:
                param = float(param)
        return param

    def set(self, name=DEFAULT_NAME, **kwargs):
        self.scores[name]['train'] = self.__get_param('train', **kwargs)
        self.scores[name]['valid'] = self.__get_param('valid', **kwargs)
        self.scores[name]['test'] = self.__get_param('test', **kwargs)
        return self.scores[name]

    def get(self, name=None):
        return self.scores[name] if name else self.scores[self.DEFAULT_NAME]

    def mean(self):
        scores = pd.DataFrame.from_dict(self.scores, orient='index')
        self.scores_mean = scores.mean(axis=0, skipna=True).to_dict()
        return self.scores_mean

    def std(self):
        scores = pd.DataFrame.from_dict(self.scores, orient='index')
        self.scores_std = scores.std(axis=0, skipna=True).to_dict()
        return self.scores_std

    def save(self):
        score_file = self.path2file / self.SCORE_FILE
        aggregate_file = self.path2file / self.AGGREGATE_FILE

        if self.prefix:
            score_file = utils.insert2filename(score_file, prefix=self.prefix)
            aggregate_file = utils.insert2filename(aggregate_file, prefix=self.prefix)

        with score_file.open(mode='w') as f:
            json.dump(self.scores, f, indent=4)

        with aggregate_file.open(mode='w') as f:
            json.dump({'means': self.scores_mean,
                        'std': self.scores_std}, f, indent=4)


class ConfusionMatrixGraph(Heatmap):
    DEFAULT_FILENAME = 'confusion_matrix.png'
    NORMALIZE_ALL = 'all'
    NORMALIZE_TRUE = 'row'
    NORMALIZE_PRED = 'column'

    def __init__(self, cm:TDATA, name=None, idx2label=None, savepath=None):
        self.idx2label = idx2label
        if idx2label:
            labels = list(idx2label.values())
            cm = pd.DataFrame(cm, index=labels, columns=labels)
        super().__init__(data=cm, name=name, savepath=savepath)
        
    def draw(self, suffix=None, normalize=NORMALIZE_TRUE, **kwargs):
        # Set title of figure
        title = f'Confusion Matrix{suffix}' if suffix else 'Confusion Matrix'

        cm = self._data.dataframe
        # normalize
        cm_normalized, cm_sum = utils.normalize_matrix(cm, normalize)

        annot = np.empty_like(cm).astype(str)
        for row, col in itertools.product(range(cm_normalized.shape[0]), range(cm_normalized.shape[1])):
            value = cm.iloc[row, col]
            normalized_value = cm_normalized[row, col]
            annot[row, col] = f'{normalized_value:.2%}\n{value}'
            if row == col:
                annot[row, col] += f'/{cm_sum[row, col]}'

        self.annot = annot
        self.data = cm_normalized
        super().draw(title=title, xlabel='Predicted labels', ylabel='True labels', **kwargs)
        self.data = cm
        return self
