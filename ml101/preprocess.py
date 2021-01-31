import pandas as pd
import numpy as np
from itertools import accumulate
from typing import Iterator, Union, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from ml101.data import Data, Types, TDATA


class BaseFilter:
    def __init__(self, data:TDATA, inplace=True, dropna=True, append=True):
        self._data = Types.check_data(data)
        self.inplace = inplace
        self.delna = dropna
        self.append = append

    def _postprocess(self, new_df:pd.DataFrame, inplace=True, append=True, axis=1, dropna=True):
        if new_df is None or new_df.empty:
            out_df = self._data._dataframe
        else:
            if append:
                if new_df.empty is False and axis == 1:
                    assert new_df.shape[0] == self._data.shape[0]
                    new_df.index = self._data.dataframe.index
                dfs = [self._data.dataframe, new_df]
                out_df = pd.concat(dfs, axis=axis)
            else:
                out_df = new_df

            if dropna:
                out_df.dropna(subset=new_df.columns, axis=0, how='all', inplace=True)

        if inplace:
            self._data._dataframe = out_df
            return_value = self
        else:
            return_value = out_df
        return return_value

    @property
    def data(self) -> Data:
        return self._data

    @data.setter
    def data(self, data:TDATA):
        self._data = Types.check_data(data)


class Insertion(BaseFilter):

    def __check4fillna(self, values):
        if isinstance(values, (int, float, str, dict)):
            return values
        if isinstance(values , Tuple):
            return {name: value for name, value in zip(*values)}
        

    def fill_na(self, values:Union[int,float,str,dict,Tuple[list,list]], inplace=None):
        values = self.__check4fillna(values)
        df = self._data.dataframe
        new_df = df.fillna(values)
        
        if inplace is None: inplace = self.inplace
        return self._postprocess(new_df, inplace=inplace, append=False, dropna=False)

    def fill_median(self, cols:list=None, inplace=None):
        df = self._data.dataframe
        if cols is None: cols = df.columns # this may be correct to select any NaN columns
        median = df[cols].median().tolist()
        return self.fill_na((cols, median), inplace=inplace)

    def fill_mean(self, cols:list=None, inplace=None):
        df = self._data.dataframe
        if cols is None: cols = df.columns  # this may be correct to select any NaN columns
        median = df[cols].mean().tolist()
        return self.fill_na((cols, median), inplace=inplace)

    def fill_max(self, cols:list=None, inplace=None):
        df = self._data.dataframe
        if cols is None: cols = df.columns  # this may be correct to select any NaN columns
        median = df[cols].max().tolist()
        return self.fill_na((cols, median), inplace=inplace)

    def fill_min(self, cols:list=None, inplace=None):
        df = self._data.dataframe
        if cols is None: cols = df.columns  # this may be correct to select any NaN columns
        median = df[cols].min().tolist()
        return self.fill_na((cols, median), inplace=inplace)

    def fill_along(self, method='ffill', axis=0, inplace=None):
        df = self._data.dataframe
        new_df = df.fillna(method=method, axis=axis)
        if inplace is None: inplace = self.inplace
        return self._postprocess(new_df, inplace=inplace, append=False, dropna=False)

    def interpolate(self):
        # TODO: to be implemented
        pass


class Removal(BaseFilter):
    def drop_na(self, axis=0, how:Union[str, int, float]='all', cols:list=None, inplace=None):
        if isinstance(how, str):
            new_df = self._data.dataframe.dropna(axis=axis, how=how, subset=cols, inplace=False)
        elif isinstance(how, int):
            new_df = self._data.dataframe.dropna(axis=axis, thresh=how, subset=cols, inplace=False)
        else:
            num = self._data.shape[0] * how
            new_df = self._data.dataframe.dropna(axis=axis, thresh=num, subset=cols, inplace=False)

        if inplace is None: inplace = self.inplace
        return self._postprocess(new_df, inplace=inplace, append=False, dropna=False)

    def drop_const(self, axis=0, dropna=False, inplace=None):
        df = self._data.dataframe
        if axis == 0:
            keep_columns = df.columns[df.nunique(dropna=dropna) > 1]
            new_df = df[keep_columns]
        else:
            keep_rows = df.index[df.nunique(axis=1, dropna=dropna) > 1]
            new_df = df.loc[keep_rows, :]

        if inplace is None: inplace = self.inplace
        return self._postprocess(new_df, inplace=inplace, append=False, dropna=False)

    def drop_columns(self, columns: list = None, inplace=None):
        new_df = self._data.dataframe.drop(columns, axis=1)

        if inplace is None: inplace = self.inplace
        return self._postprocess(new_df, inplace=inplace, append=False, dropna=False)

    def drop_rows(self, row_index: list = None, inplace=None):
        df = self._data.dataframe
        new_df = self._data.dataframe.drop(df.index[row_index])

        if inplace is None: inplace = self.inplace
        return self._postprocess(new_df, inplace=inplace, append=False, dropna=False)

    def drop_outliers(self, cols: list = None, drop_region: str='both', inplace=None):
        """Removes outliers from

        Args:
            columns (list, optional): columns to remove outloers. Defaults to None, where outliers are removed from all columns.
            drop_region (str, optional): Defaults to 'both'
                both: outliers in both lower and upper part are removed.
                lower: only the outliers in lower part are removed.
                upper: only the outliers in upper part are removed.
        """
        # TODO: to be extended to other methods
        if cols is None:
            df = self._data.dataframe
        else:
            df = self._data.dataframe[cols]

        DISCERN_CONSTANT = 1.5
        Q1 = 0.25
        Q3 = 0.75
        
        q1 = df.quantile(Q1)
        q3 = df.quantile(Q3)
        outlier_margin = (q3-q1) * DISCERN_CONSTANT
        lower = q1 - outlier_margin
        upper = q3 + outlier_margin

        # TODO: metric should be more diverse
        if drop_region == 'both':
            mask = ~((df < lower) | (df > upper)).any(axis=1)
        elif drop_region == 'lower':
            mask = ~(df < lower).any(axis=1)
        else:
            mask = ~(df > upper).any(axis=1)

        new_df = self._data.dataframe[mask]

        if inplace is None: inplace = self.inplace
        return self._postprocess(new_df, inplace=inplace, append=False, dropna=False)


class Conversion(BaseFilter):
    SCALE_STANDARD = 'standard'
    SCALE_MINMAX = 'minmax'

    def scale(self, method:str=SCALE_STANDARD, except_cols: list=None, inplace=None):
        """A function to scale the dataset

        Args:
            method (str): Scaling method (standardize or minmax scale)
            except_cols (list, optional): column names to exclude for scaling. Defaults to None.
        """
        # TODO: seperate scale methods for each
        df = self._data.dataframe

        if except_cols is None:
            except_cols = []
        
        colnames = df.columns
        colnames = colnames[~colnames.isin(except_cols)]
        
        num_cols = list()
        for col in colnames:
            if df[col].dtype in (np.int, np.float):
                num_cols.append(col)
            else:
                except_cols.append(col)
        df_num = df[num_cols]
        df_exc = df[except_cols]

        if method == self.SCALE_STANDARD:
            self.scaler_ = StandardScaler()
        elif method == self.SCALE_MINMAX:
            self.scaler_ = MinMaxScaler()
        else:
            raise ValueError('Unsupported Scaling Method!')

        df_scaled = pd.DataFrame(self.scaler_.fit_transform(df_num), index=df.index, columns=df_num.columns)
        df_scaled = pd.concat([df_scaled, df_exc], axis=1)
        df_scaled = df_scaled[df.columns]

        if inplace is None: inplace = self.inplace
        return self._postprocess(df_scaled, inplace=inplace, append=False, dropna=False)

    def onehot(self, cols:list=None, inplace=None):
        """A function to encode one-hot vector for categorical variables

        Args:
            cols (list): column names to be encoded, which are categorical

        Returns:
            pd.DataFrame: dataset after one-hot encoding
        """
        df = self._data.dataframe
        if cols is None: cols = df.columns

        encoded_cols = list()
        for col in cols:
            encoder = OneHotEncoder()
            enc_array = encoder.fit_transform(df[col].to_frame()).toarray()
            df_enc = pd.DataFrame(enc_array, index = df.index, columns=[col + '_' + str(c) for c in encoder.categories_[0]])
            encoded_cols.append(df_enc)
        
        df_encoded = pd.concat(encoded_cols, axis=1)

        if inplace is None: inplace = self.inplace
        return self._postprocess(df_encoded, inplace=inplace, append=False, dropna=False)        

    # TODO: Label binarizer should be implemented here.

    @classmethod
    def get_shifted_names(cls, columns: list, shift: int = 1):
        newnames = list()
        for colname in columns:
            suffix = f'(t{shift})' if shift < 0 else f'(t+{shift})'
            newname = colname + suffix
            newnames.append(newname)
        return newnames

    def shift(self, columns: list = None, move: int = 1, dropna=None, append=None, inplace=None) -> Union[pd.DataFrame, 'DataFilter']:
        columns = Types.check_list(columns)

        df = self._data.dataframe

        if move != 0:
            newnames = self.get_shifted_names(columns, move)
            shifted_df = df[columns].shift(move)
            shifted_df.columns = newnames
        else:
            shifted_df = pd.DataFrame()

        if dropna is None: dropna = self.delna
        if append is None: append = self.append
        if inplace is None: inplace = self.inplace
        return self._postprocess(shifted_df, inplace=inplace, append=append, dropna=dropna)

    @classmethod
    def cutoffs(cls, sr:pd.Series):
        newstart = sr.where(sr == 0).first_valid_index() if sr.iloc[0] == 1 else 0
        newend = sr.where(sr == 0).last_valid_index() if sr.iloc[-1] == 1 else (sr.size -1)
        return (newstart, newend)

    @classmethod
    def countup(cls, sr:list, labelon=1) -> Iterator:
        return accumulate(sr, lambda acc, cur: acc + 1 if cur == labelon else 0)

    @classmethod
    def period(cls, sr:list, labelon=1) -> Iterator:
        cntup = list(cls.countup(sr, labelon))
        reversed_period = list(accumulate(reversed(cntup), lambda acc, cur: max(acc, cur) if cur != 0 else 0))
        return reversed(reversed_period)

    def set_period(self, columns: list, new_columns:list=None, labelon: int = 1, append=None, inplace=None) -> Union[pd.DataFrame, 'DataFilter']:
        columns = Types.check_list(columns)

        df = self._data.dataframe
        if new_columns is None:
            new_columns = [colname + '_period' for colname in columns]

        period_df = pd.DataFrame()
        for col, newcol in zip(columns, new_columns):
            period_df[newcol] = list(self.period(df[col]))

        if append is None: append = self.append
        if inplace is None: inplace = self.inplace
        return self._postprocess(period_df, inplace=inplace, append=append, dropna=False)


class DataFilter(Insertion, Removal, Conversion):
    pass
