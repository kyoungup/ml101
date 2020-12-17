import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.pipeline import FeatureUnion
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from ml101.data import Data, Datap

class SimpleTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        print('I am Simple')

    def fit(self, x, y=None):
        print(f'step1 (fit) - {type(x)}')
        return self

    def transform(self, x):
        print(f'step1 (transform) - {type(x)}')
        return x

    def inverse_transform(self, x):
        return x

class SimpleTransformer2(BaseEstimator, TransformerMixin):
    def __init__(self):
        print('I am Simple2')

    def fit(self, x, y=None):
        print(f'step2 (fit) - {type(x)}')
        return self

    def transform(self, x):
        print(f'step2 (transform) - {type(x)}')
        return x

    def inverse_transform(self, x):
        return x

if __name__ == '__main__':
    datap = Datap(pd.DataFrame({#'pet':      ['cat', 'dog', 'dog', 'fish', 'cat', 'dog', 'cat', 'fish'],
                         'children': [4., 6, 3, 3, 2, 3, 5, 4],
                         'salary':   [90., 24, 44, 27, 32, 59, 36, 27]})
    )
    
    data = Data(datap)

    simple_pipeline = Pipeline([
        ('step1', SimpleTransformer()),
        ('step1.5', StandardScaler()),
        ('step2', SimpleTransformer2()),
        ('step3', SimpleTransformer2())
        ])
    result = simple_pipeline.fit(data)
    print(result)
    
    