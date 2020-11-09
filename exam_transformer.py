import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.pipeline import FeatureUnion
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from uids.data import Data, Datap, DataMixin

class SimpleTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        print('I am Simple')

    def fit(self, X, y=None):
        print(f'step1 (fit) - {type(X)}')
        return self

    def transform(self, X):
        print(f'step1 (transform) - {type(X)}')
        return X

    def inverse_transform(self, X):
        return X

class SimpleTransformer2(BaseEstimator, TransformerMixin):
    def __init__(self):
        print('I am Simple2')

    def fit(self, X, y=None):
        print(f'step2 (fit) - {type(X)}')

        return self

    def transform(self, X):
        print(f'step2 (transform) - {type(X)}')
        return X

    def inverse_transform(self, X):
        return X

if __name__ == '__main__':
    datap = Datap(pd.DataFrame({#'pet':      ['cat', 'dog', 'dog', 'fish', 'cat', 'dog', 'cat', 'fish'],
                         'children': [4., 6, 3, 3, 2, 3, 5, 4],
                         'salary':   [90., 24, 44, 27, 32, 59, 36, 27]})
    )
    
    data = Data(datap)

    simple_pipeline = Pipeline([
        ('step1', SimpleTransformer()),
        ('step1.5', DataMixin(StandardScaler())),
        ('step2', SimpleTransformer2()),
        ('step3', SimpleTransformer2())
        ])
    result = simple_pipeline.fit_transform(data)
    print(result.dataframe)
    
    