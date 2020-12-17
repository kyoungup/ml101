import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours, TomekLinks
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from collections import OrderedDict
from collections import Counter


class Sampler:
    # sampling strategies (method) to choose
    MINOR = 'minority' # resample only the minority class
    NOT_MINOR = 'not minority' # resample all classes but the minority class
    NOT_MAJOR = 'not majority' # resample all classes but the majority class
    ALL = 'all' # resample all classes
    AUTO = 'auto'   # equivalent to NOT_MAJOR (oversampling) or NOR_MINOR (undersampling)
    # or ratio can be a dict to control each, i.e., {label: #samples}

    def __init__(self, features: pd.DataFrame, y: pd.Series, random_state=None, replacement=False):
        self.features = features
        self.y = y if isinstance(y, pd.Series) else pd.Series(y)
        self.random_state = random_state
        self.replacement = replacement
        self.sample_indices = None

    def __trace_previous_indices(self, indices):
        if indices is not None:
            if self.sample_indices:
                return self.sample_indices[indices]
            else:
                return indices
        return None

    def smote(self, method=AUTO):
        sampler = SMOTE(sampling_strategy=method, random_state=self.random_state)
        X, y = sampler.fit_resample(self.features, self.y)
        self.features = pd.DataFrame(X, columns=self.features.columns)
        self.y = pd.Series(y, name=self.y.name)
        self.sample_indices = None
        return self.features, self.y

    def smoteenn(self, method=AUTO):
        sampler = SMOTEENN(sampling_strategy=method, random_state=self.random_state)
        X, y = sampler.fit_resample(self.features, self.y)
        self.features = pd.DataFrame(X, columns=self.features.columns)
        self.y = pd.Series(y, name=self.y.name)
        self.sample_indices = None
        return self.features, self.y

    def smotetomek(self, method=AUTO):
        sampler = SMOTETomek(sampling_strategy=method, random_state=self.random_state)
        X, y = sampler.fit_resample(self.features, self.y)
        self.features = pd.DataFrame(X, columns=self.features.columns)
        self.y = pd.Series(y, name=self.y.name)
        self.sample_indices = None
        return self.features, self.y

    def oversample(self, method=AUTO):
        # define undersample strategy
        sampler = RandomOverSampler(sampling_strategy=method, random_state=self.random_state)
        # fit and apply the transform
        X, y = sampler.fit_resample(self.features, self.y)
        self.features = pd.DataFrame(X, columns=self.features.columns)
        self.y = pd.Series(y, name=self.y.name)
        self.sample_indices = self.__trace_previous_indices(sampler.sample_indices_)
        return self.features, self.y

    def undersample(self, method=AUTO):
        # define undersample strategy
        sampler = RandomUnderSampler(sampling_strategy=method, random_state=self.random_state,
                                    replacement=self.replacement)
        # fit and apply the transform
        X, y = sampler.fit_resample(self.features, self.y)
        self.features = pd.DataFrame(X, columns=self.features.columns)
        self.y = pd.Series(y, name=self.y.name)
        self.sample_indices = self.__trace_previous_indices(sampler.sample_indices_)
        return self.features, self.y

    def enn(self, method=AUTO):  
        # define undersample strategy
        sampler = EditedNearestNeighbours(sampling_strategy=method)
        # fit and apply the transform
        X, y = sampler.fit_resample(self.features, self.y)
        self.features = pd.DataFrame(X, columns=self.features.columns)
        self.y = pd.Series(y, name=self.y.name)
        self.sample_indices = self.__trace_previous_indices(sampler.sample_indices_)
        return self.features, self.y

    def tomek(self, method=AUTO):  
        # define undersample strategy
        sampler = TomekLinks(sampling_strategy=method)
        # fit and apply the transform
        X, y = sampler.fit_resample(self.features, self.y)
        self.features = pd.DataFrame(X, columns=self.features.columns)
        self.y = pd.Series(y, name=self.y.name)
        self.sample_indices = self.__trace_previous_indices(sampler.sample_indices_)
        return self.features, self.y

    def custom(self, methods=OrderedDict(smote=AUTO, enn=AUTO)):
        for sampler, ratio in methods.items():
            if sampler == 'under':
                self.undersample(ratio)
            elif sampler == 'over':
                self.oversample(ratio)
            elif sampler == 'smote':
                self.smote(ratio)
            elif sampler == 'smoteenn':
                self.smoteenn(ratio)
            elif sampler == 'smotetomek':
                self.smotetomek(ratio)
            elif sampler == 'enn':
                self.enn(ratio)
            elif sampler == 'tomek':
                self.tomek(ratio)
            
        return self.features, self.y

    def resample(self, n_samples, random_state=None, replace=None):
        if random_state is None: random_state = self.random_state
        if replace is None: replace = self.replacement
        sampled_array = resample(self.dataframe, n_samples=n_samples, stratify=self.y, replace=replace, random_state=random_state)
        self.features = pd.DataFrame(sampled_array.iloc[:, 0:-1], columns=self.features.columns)
        self.y = pd.Series(sampled_array.iloc[:, -1], name=self.y.name)
        self.sample_indices = None
        return self.features, self.y

    @property
    def dataframe(self):
        return pd.concat([self.features, self.y], axis=1)

    @property
    def dataset(self):
        return self.features, self.y
    
    @property
    def count(self):
        # self.y.value.value_count() can be used
        return Counter(self.y.to_numpy())

    def fit(self):
        return self

    def tansform(self):
        pass