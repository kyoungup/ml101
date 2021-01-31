import pandas as pd
import numpy as np
from scipy import stats
from typing import Union
from ml101.data import Types, TDATA
from ml101.graphs import Heatmap, Point
import ml101.utils as utils


class Correlation:
    # Methods
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"

    DEFAULT_NAME = 'CORR'

    def __init__(self, data:TDATA):
        self.data = Types.check_data(data)
        self.corr_ = None
        self.graph = None
        self.name = None

    def infer(self, col:str=None, group=None, method:str=PEARSON) -> 'Correlation':
        """calculate correlation coefficient

        Args:
            y (str, optional): a single columns name to get correlation with. Defaults to None.
            group (str, optional): a single column name of group value. Defaults to None.
            path (Path, optional): path to save heatmap. Defaults to None.

        Returns:
            Series: result of correlation coefficient in order
        """
        df = self.data.dataframe
        if col is not None:
            self.corr_ = df.corrwith(df[col], method=method)
            self.corr_ = self.corr_.drop(col)
        else:
            self.corr_ = df.corr(method=method)

        return self

    @property
    def corr(self) -> Union[pd.DataFrame, pd.Series]:
        return self.corr_

    def sort(self, n=1, abs=True, ascending=False) -> pd.Series:
        if isinstance(self.corr_, pd.Series):
            self.sorted_corr_ = self.corr_
        else:
            self.sorted_corr_ = self.corr_.where(np.triu(np.ones(self.corr_.shape), k=n).astype(np.bool)).stack()

        if abs:
            self.sorted_corr_ =  self.sorted_corr_.abs()
        self.sorted_corr_ = self.sorted_corr_.sort_values(ascending=ascending)
        return self.sorted_corr_

    def show(self, name:str=None, **kwargs):
        if isinstance(self.corr_, pd.DataFrame):
            self.name = name if name else self.DEFAULT_NAME
            self.graph = Heatmap(data=self.corr_, name=self.name)
            self.graph.draw(title=self.name, **kwargs)
        return self

    def save(self, save_path:str):
        if isinstance(self.corr_, pd.DataFrame):
            assert save_path
            save_path, filename = utils.confirm_path(save_path)
            if filename is None: filename = self.name + f'_{self.DEFAULT_NAME.lower()}_clusters.png'
            return self.graph.save(save_path / filename)
        return None


class MeanComparison:
    def __init__(self, data:TDATA):
        self.data = Types.check_data(data)

    def mean_comparison(self, group, col) -> float:
        df = self.data.dataframe
        freq = df[group].value_counts()
        group_values = freq[freq > 2].index.values
        n_groups = len(group_values)

        if n_groups < 2:
            raise ValueError('Less than 2 Groups')
        elif n_groups == 2:
            g0 = df[col][df[group] == group_values[0]]
            g1 = df[col][df[group] == group_values[1]]

            _, p_val = stats.levene(g0, g1)

            if p_val < 0.05:
                _, p_value = stats.ttest_ind(g0, g1, equal_var=False)
            else:
                _, p_value = stats.ttest_ind(g0, g1, equal_var=True)
            
            return p_value
        else:
            grouped_cols = [ df[col][df[group] == grp] for grp in group_values ]
            _, p_value = stats.f_oneway(*grouped_cols)
        return p_value

    def siginificant_variables(self, group:str, cols:list=None, threshold=0.05) -> pd.Series:
        """Derive variables with significantly different means between groups

        Args:
            data (pandas.DataFrame): Data to analyze
            group (str): column name from data to be used as group
            variables (list, optional): list of column names to be checked. Default is all numeric columns
            threshold (float, optional): p-value criteria for significant variable selection. Defaults to 0.05.
        Returns:
            vars (list): column list with significantly different means ordered by p-value
        """
        if cols:
            numeric_cols = pd.Index(cols)
        else:
            numeric_cols = self.data.numeric_cols
        numeric_cols = numeric_cols.drop(group, errors='ignore')

        p_values = numeric_cols.to_series(index=numeric_cols).apply(lambda col: self.mean_comparison(group, col))
        print(f'Median Value for All p-values : {p_values.median()}')
        sig_list = p_values.loc[p_values < threshold]
        sig_list.sort_values(inplace=True)

        print(sig_list)
        return sig_list


class Interval(Point):
    DEFAULT_FILENAME = 'interval.png'
    # default style
    DEFAULT_STYLE = dict(scale=0.8,
                         capsize=0.05,
                         errwidth=2)

    def __init__(self, data:TDATA, x=None, y=None, group=None, size=None,
                 confidence=95, join=True, ax=None, name=None, savepath=None):
        super().__init__(data=data, x=x, y=y, group=group, join=join, confidence=confidence, ax=ax, name=name, savepath=savepath)

    @classmethod
    def mean_interval(cls, vec:Union[pd.Series, np.array], confidence:float=95) -> list:
        if confidence > 1: confidence /= 100.0
        m, se = np.mean(vec), stats.sem(vec)
        h = se * stats.t.ppf((1 + confidence) / 2, len(vec)-1)
        return m, m-h, m+h

    @classmethod
    def calc_intervals(cls, data:pd.DataFrame, group:str, variable:str, confidence:float=95) -> dict:
        freq = data[group].value_counts(sort=False)
        group_info = freq[freq > 2].index.values

        intervals = {'category': ['mean', 'lower', 'upper']}
        for gr in group_info:
            temp = data[data[group] == gr][variable]
            intervals[gr] = list(cls.mean_interval(temp, confidence))

        return intervals

    def draw(self, ax=None, x=None, y=None, group=None, title=None, xlabel=None, ylabel=None, join=None, confidence=None, **kwargs):
        self.kwargs_ = self.DEFAULT_STYLE.copy()
        self.kwargs_.update(kwargs)
        if confidence is None: confidence = self.confidence
        # Set title of figure
        title = f'{confidence}% Confidence Interval Plot'
        super().draw(ax=ax, x=x, y=y, group=group, title=title, xlabel=xlabel, ylabel=ylabel, join=join, confidence=confidence, **kwargs)
        return self
