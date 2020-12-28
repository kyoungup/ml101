import pandas as pd
from itertools import accumulate
from typing import Iterator

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from ml101.data import Data


class Shift(BaseEstimator, TransformerMixin):
    def __init__(self, columns:list = None, move:int = 1, dropna=True, append=True, inplace=True):
        self.columns = columns
        self.move = move
        self.dropna = dropna
        self.append = append
        self.inplace = inplace

    def fir(self, X, y=None):
        # X = check_array(X, accept_sparse=True)
        self.n_features_ = X.shape[1]
        # Return the transformer
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # X = super().transform(X)
        if self.columns is None:
            self.columns = X.columns
        return self.shift(Data(X), self.columns, self.move, self.dropna, self.append, self.inplace)

    @classmethod
    def getshiftednames(cls, columns:list = None, shift:int = 1):
        newnames = list()
        for colname in columns:
            suffix = f'(t{shift})' if shift < 0 else f'(t+{shift})'
            newname = colname + suffix
            newnames.append(newname)
        return newnames

    @classmethod
    def shift(cls, data:Data, columns:list = None, move:int = 1, dropna=True, append=True, inplace=True) -> Data:
        assert isinstance(columns, list)
        df = data._dataframe

        if move != 0:
            newnames = cls.getshiftednames(columns, move)
            shifted_df = df[columns].shift(move)
            shifted_df.columns = newnames
        else:
            shifted_df = pd.DataFrame()

        if append:
            out_df = pd.concat([df, shifted_df], axis=1)
        else:
            out_df = shifted_df

        if inplace:
            data.dataframe = out_df
        else:
            data = Data(df)

        if dropna:
            data.dataframe.dropna(inplace=True)

        data.dataframe.reset_index(drop=True, inplace=True)

        return data

    def _get_selection(self, df):
        assert isinstance(df, pd.DataFrame)
        return df[self.columns]


class SetPeriod(BaseEstimator, TransformerMixin):
    def __init__(self, columns:list=None, outs:list=None, labelon=1, append=True, inplace=True):
        self.columns = columns
        self.outs = outs
        self.labelon = labelon
        self.append = append
        self.inplace = inplace

    def fit(self, X, y=None):
        self.n_features_ = X.shape[1]
        return self

    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        outs = list()
        for col, out in zip(self.columns, self.outs):
            outs.append(pd.DataFrame(self.period(X[col], self.labelon), columns=[out]))

        if self.append:
            result = pd.concat([X] + outs, axis=1)
        else:
            result = pd.concat(outs, axis=1)
        # result = Data(result)
        return result

    @classmethod
    def countup(cls, sr:list, labelon=1) -> Iterator:
        return accumulate(sr, lambda acc, cur: acc + 1 if cur == labelon else 0)

    @classmethod
    def period(cls, sr:list, labelon=1) -> Iterator:
        cntup = list(cls.countup(sr, labelon))
        reversed_period = list(accumulate(reversed(cntup), lambda acc, cur: max(acc, cur) if cur != 0 else 0))
        return reversed(reversed_period)


class Preprocessor:
    def __init__(self, data:pd.DataFrame, inplace=True):
        self.data = data
        self.inplace = inplace

    def fill(self, cols):
        pass

    def fillmedian(self, cols:list):
        df = self.data.dataframe
        median = df[cols].median().tolist()
        self.fillna(cols, median)

    def fillna(self, cols: list, values: list, colvals: dict = None):
        df = self.data.dataframe
        if colvals is not None:
            for col, value in colvals.items():
                df[col].fillna(value, inplace=True)
        else:
            if len(values) > 1:
                assert len(cols) == len(values)
                for idx, value in enumerate(values):
                    df[cols[idx]].fillna(value, inplace=True)
            else:
                df[cols[0]].fillna(values[0], inplace=True)

    def dropna(self):
        self.data.dataframe.drop(inplace=self.inplace)
        return self

    def dropconst(self):
        df = self.data.dataframe
        keep_columns = df.columns[df.nunique() > 1]
        self.data.dataframe = df[keep_columns]
        return self

    def drop_columns(self, columns: list = None):
        self.data.datafreame.drop(columns, axis=1, inplace=self.inpalce)
        return self

    def drop_rows(self, row_index: list = None):
        df = self.data.dateframe
        self.data.dataframe.drop(df.index[row_index], inplace=self.inplace)
        return self

    def drop_outliers(self, columns: list = None, drop_region: str='both'):
        """Removes outliers from

        Args:
            columns (list, optional): columns to remove outloers. Defaults to None, where outliers are removed from all columns.
            drop_region (str, optional): Defaults to 'both'
                both: outliers in both lower and upper part are removed.
                lower: only the outliers in lower part are removed.
                upper: only the outliers in upper part are removed.
        """
        if columns is None:
            columns = self.data.dataframe.columns

        DISCERN_CONSTANT = 1.5
        Q1 = 0.25
        Q3 = 0.75

        df = self.data.dataframe[columns]
        df.dropna(inplace=self.inplace)
        q1 = df.quantile(Q1)
        q3 = df.quantile(Q3)
        outlier_margin = (q3-q1) * DISCERN_CONSTANT
        lower = q1 - outlier_margin
        upper = q3 + outlier_margin
        if drop_region == 'both':
            mask = (df < upper) & (df > lower)
        elif drop_region == 'lower':
            mask = df > lower
        else:
            mask = df < upper

        df = df[mask]
        self.data.dataframe = self.data.dataframe.iloc[df.index]
        return self

    def getshiftednames(self, columns: list = None, shift: int = 1):
        newnames = list()
        for colname in columns:
            suffix = f'(t{shift})' if shift < 0 else f'(t+{shift})'
            newname = colname + suffix
            newnames.append(newname)
        return newnames

    def shift(self, columns: list = None, shift: int = 1, dropna=True, append=True, inplace=True):
        assert isinstance(columns, list)

        if append:
            out_df = pd.DataFrame(self.data.dataframe)
        else:
            out_df = pd.DataFrame()

        if shift != 0:
            df = self.data.dataframe
            newnames = self.getshiftedname(columns, shift)
            for colname, newname in zip(columns, newnames):
                out_df[newname] = df[colname].shift(shift)
            
            if dropna:
                out_df = out_df.dropna().reset_index()
        if inplace:
            self.data.dataframe = out_df
        
        return out_df

    @classmethod
    def cutoffs(cls, sr:pd.Series):
        newstart = sr.where(sr == 0).first_valid_index() if sr.iloc[0] == 1 else 0
        newend = sr.where(sr == 0).last_valid_index() if sr.iloc[-1] == 1 else (sr.size -1)

        return (newstart, newend)


class Conversion:
    def __init__(self, data: pd.DataFrame):
        self.dataset = data

    def scale(self, method, except_cols: list=None):
        """A function to scale the dataset

        Args:
            method (str): Scaling method (standardize or minmax scale)
            except_cols (list, optional): column names to exclude for scaling. Defaults to None.
        """
        if except_cols is None:
            except_cols = []

        colname = self.dataset.columns

        for col in colname:
            if self.dataset[col].dtype == 'int64' or self.dataset[col].dtype == 'float64':
                pass
            else:
                if except_cols.count(col) == 0:
                    except_cols.append(col)

        if len(except_cols) > 0:
            df_num = self.dataset.loc[:, ~self.dataset.columns.isin(except_cols)]
            df_exc = self.dataset.loc[:, self.dataset.columns.isin(except_cols)]
        else:
            df_num = self.dataset.copy()
            df_exc = pd.DataFrame(index=self.dataset.index)

        if method == 'Standard':
            scaler = StandardScaler()
        elif method == 'MinMax':
            scaler = MinMaxScaler()
        else:
            raise ValueError('Unsupported Scaling Method!')

        df_scaled = pd.DataFrame(scaler.fit_transform(df_num), index=self.dataset.index, columns=df_num.columns)
        df_scaled = pd.concat([df_scaled, df_exc], axis=1)
        df_scaled = df_scaled[colname]

        return df_scaled, scaler

    def onehot(self, categorical_variables):
        """A function to encode one-hot vector for categorical variables

        Args:
            categorical_variables (list): column names to be encoded

        Returns:
            pd.DataFrame: dataset after one-hot encoding
        """
        df_encoded = self.dataset.copy()
        for cat in categorical_variables:
            enc = OneHotEncoder()
            df_enc = enc.fit_transform(self.dataset[cat].to_frame()).toarray()
            df_enc = pd.DataFrame(df_enc, index = self.dataset.index, columns=[cat + '_' + str(c) for c in enc.categories_[0]])
            df_encoded = pd.concat([df_encoded, df_enc], axis=1)

        return df_encoded