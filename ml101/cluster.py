from abc import ABCMeta, abstractmethod, abstractproperty
import sklearn.cluster as algs
import numpy as np
from pathlib import Path
from ml101.data import Data, Types, TDATA, TLT, TAR
from ml101.graphs import Scatter
import ml101.utils as utils


class Cluster(metaclass=ABCMeta):
    DEFAULT_N_CLUSTERS = 3
    
    def __init__(self, data:TDATA, cols:TLT=None, n_clusters:int=None):
        self.data = Types.check_data(data)
        self.cols = Types.check_list(cols) if cols else Types.check_list(self.data.columns)
        self.n_clusters = n_clusters if n_clusters else self.DEFAULT_N_CLUSTERS

        self.alg = None
        self.new_features_ = None
        self.labels_ = None

    @abstractproperty
    def centers(self) -> np.ndarray:
        if hasattr(self.alg, 'cluster_centers_'):
            return self.alg.cluster_centers_
        else:
            return None

    @property
    def labels(self) -> list:
        return self.labels_

    @property
    def ndim(self) -> int:
        if self.cols:
            return len(self.cols)
        else:
            return None

    @abstractmethod
    def cluster(self, cols:TLT=None, **kwargs) -> 'Cluster':
        cols = Types.check_list(cols) if cols else self.cols
        df = self.data.dataframe[cols]
        self.alg.fit(df, **kwargs)
        return self

    def _check_fit(self, features:TDATA) -> Data:
        if self.centers is None:
            raise AssertionError('cluster() should be called before this call')
        if features is None:
            features = self.data.dataframe[self.cols]
        else:
            features = Types.check_data(features)
            assert features.shape[1] == len(self.cols)
        return features

    def predict(self, features:TDATA=None) -> list:
        features = self._check_fit(features)
        self.labels_ = self.alg.predict(features).tolist()
        return self.labels_

    def transform(self, features) -> np.ndarray:
        features = self._check_fit(features)
        self.new_features_ = self.alg.transform(features)
        return self.new_features_

    @abstractmethod
    def show(self, x:str, y:str, mark:bool=True, name:str=None) -> 'Cluster':
        pass
    
    def save(self, save_path:str) -> Path:
        assert save_path
        save_path, filename = utils.confirm_path(save_path)
        if filename is None: filename = self.name + f'_{self.DEFAULT_NAME.lower()}_clusters.png'
        return self.graph.save(save_path / filename)


class KMeans(Cluster):
    DEFAULT_NAME = 'KMeans'

    def __init__(self, data:TDATA, cols:TLT=None, n_clusters:int=None, method=None):
        super().__init__(data, cols, n_clusters)
        self.alg = algs.KMeans(n_clusters=self.n_clusters)
        self.name = None

    @property
    def centers(self) -> np.ndarray:
        return super().centers

    def cluster(self, cols: TLT = None, sample_weight:TAR=None,
                n_clusters=None, init='k-means++', n_init=10, max_iter=300,
                tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='auto'):
        
        if n_clusters is None: n_clusters = self.n_clusters
        self.kwargs_ = dict(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol,
                            verbose=0, random_state=random_state, copy_x=copy_x, algorithm=algorithm)
        self.alg.set_params(**self.kwargs_)

        super().cluster(cols, sample_weight=sample_weight)
        return self

    def show(self, x:str, y:str, mark:bool=True, name:str=None):
        self.name = name if name else self.DEFAULT_NAME
        
        group = self.labels_ if mark else None
        self.graph = Scatter(name=self.name, data=self.data, x=x, y=y, group=group)
        self.graph.draw()
        return self


# class MeanShift(Cluster):
#     def cluster(self, cols:TLT=None,
#                 bandwidth: float = None, seeds: TAR = None,
#                 bin_seeding: bool = False, min_bin_freq: int = 1,
#                 cluster_all: bool = True, n_jobs: int = None, max_iter: int = 300):
#         self.kwargs_ = dict(bandwidth=bandwidth, seeds=seeds,
#                             bin_seeding=bin_seeding, min_bin_freq=min_bin_freq,
#                             cluster_all=cluster_all, n_jobs=n_jobs, max_iter=max_iter)
#         self.alg.set_params(**self.kwargs_)
#         super().cluster(cols, **self.kwargs_)
#         return self
