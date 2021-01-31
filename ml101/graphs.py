from abc import ABCMeta, abstractmethod
import numpy as np
import seaborn as sns
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import math
from scipy import stats
from typing import Union
import itertools
import ml101.utils as utils
from ml101.data import Types, TDATA


class Graph(metaclass=ABCMeta):
    # Relational plot
    SCATTER = 'scatter'    
    LINE = 'line'
    # Distribution plot
    HISTOGRAM = 'hist'
    KDE = 'kde'
    ECDF = 'ecdf'
    RUG = 'rug'
    # Categorical plot
    STRIP = 'strip'
    SWARM = 'swarm'
    BOX = 'box'
    VIOLIN = 'violin'
    BOXENV = 'boxen'
    POINT = 'point'
    BAR = 'bar'
    COUNT = 'count'
    # Etc.
    HEATMAP = 'heatmap'

    # Title Attr
    TITLE_SIZE = 'medium'
    TITLE_WEIGHT = 'normal'

    DEFAULT_FILENAME = 'graph.png'

    def __init__(self, data:TDATA=None, kind=None, name=None, ax=None, savepath=None):
        self._data = Types.check_data(data)
        self.kind = kind
        self.parent = None
        
        self.name = name
        self.fig = None
        self.axes = None
        self.ax = ax
        self.savepath = None
        self.filename = None
        self._parse_savepath(savepath)
        
    def _parse_savepath(self, savepath:str=None):
        if savepath:
            savepath = Path(savepath)
            if savepath.suffix != '':   # to check if it is a file
                self.savepath = savepath.parent
                self.filename = savepath.name
            else:
                self.savepath = savepath
                self.filename = utils.insert2filename(self.DEFAULT_FILENAME, prefix=f'{utils.convert2filename(self.name)}_')

    def _check_inputs(self, data:pd.DataFrame, x, y, group, size) -> tuple:
        # only if a new set is given
        if x is None and y is None and group is None and size is None:
            x, y, group, size = self.x, self.y, self.group, self.size

        # group(hue), size, and etc. are not used in wide form
        if data is not None and (isinstance(x, str) and isinstance(y, str)) or\
                (isinstance(x, str) and y is None) or (x is None and isinstance(y, str)):
            return (data, x, y, group, size)
        elif data is not None and x is None and y is None:
            return (data, None, None, None, None)
        elif data is not None and x is None and isinstance(y, list):
            return (data[y], None, None, None, None)
        elif data is not None and isinstance(x, list) and y is None:
            return (data[x], None, None, None, None)
        elif data is not None and isinstance(x, str) and isinstance(y, list):
            return (data.set_index(x)[y], None, None, None, None)
        elif isinstance(x, pd.Series) and isinstance(y, pd.Series) and len(x) > 1 and len(y) > 1 and len(x) == len(y):
            return (None, x, y, None, None)
        else:
            raise ValueError('Seaborn cannot process your inputs!')

    def _post_process(self, fig=None, ax=None, title=None, xlabel=None, ylabel=None, close=True):
        if fig is None:
            if ax:
                fig = ax.get_figure()
            else:
                fig = self.fig
                ax = fig
        else:
            ax = fig
        self.set_title(ax, title)
        self.set_labels(ax, xlabel, ylabel)
        fig.autofmt_xdate()
        fig.tight_layout()

        if close:
            plt.close()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data:TDATA):
        self._data = Types.check_data(data)

    @abstractmethod
    def draw(self, **kwargs):
        pass

    def show(self):
        plt.show()
        return self

    def set_title(self, fig, title:str=None):
        if title is None and self.name is None:
            return
        
        title = title.upper() if title else self.name.upper()
        fig.set_title(title, fontsize=self.TITLE_SIZE, fontweight=self.TITLE_WEIGHT)

    def set_labels(self, fig, xaxis=None, yaxis=None):
        if xaxis:
            fig.set_xlabel(xaxis, fontsize=self.TITLE_SIZE, fontweight=self.TITLE_WEIGHT)
        if yaxis:
            fig.set_ylabel(yaxis, fontsize=self.TITLE_SIZE, fontweight=self.TITLE_WEIGHT)

    def save(self, savepath=None):
        self._parse_savepath(savepath)
        if self.savepath and self.filename:
            filepath = self.savepath / self.filename
        else:
            raise ValueError('No savepath is given.')

        if self.ax:
            self.ax.get_figure().savefig(filepath)
        elif self.fig:
            self.fig.savefig(filepath)
        return filepath

    @classmethod
    def make_graphs(name, data:TDATA, kind, xs:list=None, ys:list=None) -> list:
        data = Types.check_dataframe(data)
        list_ys = ys if ys else data.columns.to_list()
        list_xs = xs if xs else [None]
        return [dict(kind=kind, name=name, x=x, y=y, data=data) for x, y in zip(itertools.cycle(list_xs), list_ys)]
        

class Canvas(Graph):
    TITLE_SIZE = 'large'
    TITLE_WEIGHT = 'bold'
    DEFAULT_FILENAME = 'canvas.png'

    def __init__(self, name=None, data:TDATA=None, savepath=None):
        super().__init__(data=data, name=name, savepath=savepath)
        self._graphs = list()

    def add(self, graphs:Union[list, Graph, 'Canvas']):
        if isinstance(graphs, Canvas):
            graphs = graphs._graphs.copy()

        for graph in graphs:
            if isinstance(graph, dict):
                graph = self.convert(**graph)
                self._graphs.append(graph)
            elif isinstance(graph, Graph):
                self._graphs.append(graph)
        return self
    
    def remove(self, indices:list):
        for idx in indices:
            del self._graphs[idx]
            del self.axes[idx]
        return self

    def convert(self, **kwargs):
        if 'kind' in kwargs:
            kind = kwargs.pop('kind')
            if 'data' not in kwargs: kwargs['data'] = self._data
            if kind == Graph.SCATTER:
                return Scatter(**kwargs)
            elif kind == Graph.COUNT:
                return Count(**kwargs)
            elif kind == Graph.LINE:
                return Line(**kwargs)
            else:
                raise ValueError(f'Unsupported Graph Type {kind}')
        else:
            # TODO: automatically determin types and assign them
            raise ValueError('Graph type is missing')

    def set_title(self, _, title:str=None):
        if title is None and self.name is None:
            return
        
        title = title.upper() if title else self.name.upper()
        # TODO: set y position
        self.fig.suptitle(title, fontsize=self.TITLE_SIZE, fontweight=self.TITLE_WEIGHT)

    def draw(self, col_wrap:int=None, sharex=False, sharey=False, title=None, figsize:tuple=None, **kwargs):
        num_graphs = len(self._graphs)
        num_cols = num_graphs if col_wrap is None or col_wrap >= num_graphs else col_wrap
        num_rows = math.ceil(num_graphs / num_cols)
        self.fig, self.axes = plt.subplots(ncols=num_cols, nrows=num_rows,
                                            sharex=sharex, sharey=sharey,
                                            squeeze=False, figsize=figsize)
        num_axes = num_cols * num_rows                                           
        if num_graphs != num_axes:
            for idx in range(num_graphs, num_axes):
                self.fig.delaxes(self.axes.flat[idx])

        for idx, graph in enumerate(self._graphs):
            graph.draw(ax=self.axes.flat[idx], **kwargs)

        self._post_process(fig=self.fig, title=title, xlabel=None, ylabel=None, close=False)
        return self

    def save(self, savepath=None, each=False):
        path_canvas = super().save(savepath)

        file_graphs = None
        if each:
            plt.close()
            file_graphs = list()
            if savepath is None:
                savepath = path_canvas.parent
            for idx, graph in enumerate(self._graphs):
                file_graph = savepath / utils.insert2filename(path_canvas.name, suffix=f'_{idx}')
                file_graphs.append(file_graph)
                graph.draw().save(file_graph)
            return path_canvas, file_graphs
        return path_canvas
        

class Scatter(Graph):
    DEFAULT_FILENAME = 'scatter.png'

    def __init__(self, data:TDATA, x=None, y=None,
                 group=None, size=None, ax=None, name=None, savepath=None):
        super().__init__(data=data, kind=Graph.SCATTER, name=name, ax=ax, savepath=savepath)
        self.x = x
        self.y = y
        self.group = group
        self.size = size
        # setattr more

    def draw(self, ax=None, x=None, y=None, group=None, size=None, title=None, xlabel=None, ylabel=None, **kwargs):
        data, x, y, group, size = self._check_inputs(self._data.dataframe, x, y, group, size)
        self.kwargs_ = kwargs.copy()
        self.kwargs_.update(data=data, x=x, y=y, hue=group, size=size, ax=ax)
        
        self.ax = sns.scatterplot(**self.kwargs_)

        self._post_process(ax=self.ax, title=title, xlabel=xlabel, ylabel=ylabel, close=ax is None)
        return self


class Line(Graph):
    DEFAULT_FILENAME = 'line.png'

    def __init__(self, data:TDATA, x=None, y=None,
                 group=None, size=None, ax=None, name=None, savepath=None):
        super().__init__(data=data, kind=Graph.LINE, name=name, ax=ax, savepath=savepath)
        self.x = x
        self.y = y
        self.group = group
        self.size = size
        # setattr more

    def draw(self, ax=None, x=None, y=None, group=None, size=None, title=None, xlabel=None, ylabel=None, **kwargs):
        data, x, y, group, size = self._check_inputs(self._data.dataframe, x, y, group, size)
        self.kwargs_ = kwargs.copy()
        self.kwargs_.update(data=data, x=x, y=y, hue=group, size=size, ax=ax)

        self.ax = sns.lineplot(**self.kwargs_)

        self._post_process(ax=self.ax, title=title, xlabel=xlabel, ylabel=ylabel, close=ax is None)
        return self


class Count(Graph):
    DEFAULT_FILENAME = 'count.png'

    def __init__(self, data:TDATA, x=None, y=None,
                 group=None, ax=None, name=None, savepath=None):
        super().__init__(data=data, kind=Graph.COUNT, name=name, ax=ax, savepath=savepath)
        self.x = x
        self.y = y
        self.group = group
        self.size = None
        # setattr more

    def draw(self, ax=None, x=None, y=None, group=None, title=None, xlabel=None, ylabel=None, **kwargs):
        data, x, y, group, _ = self._check_inputs(self._data.dataframe, x, y, group, None)
        self.kwargs_ = kwargs.copy()
        self.kwargs_.update(data=data, hue=group, ax=ax, x=x, y=y)

        self.ax = sns.countplot(**self.kwargs_)
        
        self._post_process(ax=self.ax, title=title, xlabel=xlabel, ylabel=ylabel, close=ax is None)
        return self


class Histogram(Graph):
    DEFAULT_FILENAME = 'histogram.png'

    def __init__(self, data:TDATA, x=None, y=None, group=None,
                 kernel_density=False, log_scale=False, ax=None, name=None, savepath=None):
        super().__init__(data=data, kind=Graph.HISTOGRAM, name=name, ax=ax, savepath=savepath)
        self.x = x
        self.y = y
        self.group = group
        self.size = None
        self.kernel_density = kernel_density
        self.log_scale = log_scale

    def draw(self, ax=None, x=None, y=None, group=None, kenel_density=None, log_scale=None, title=None, xlabel=None, ylabel=None, **kwargs):
        data, x, y, group, _ = self._check_inputs(self._data.dataframe, x, y, group, None)
        if kenel_density is None: kenel_density = self.kernel_density
        if log_scale is None: log_scale = self.log_scale
        self.kwargs_ = kwargs.copy()
        self.kwargs_.update(data=data, hue=group, ax=ax, x=x, y=y, kde=kenel_density, log_scale=log_scale)

        self.ax = sns.histplot(**self.kwargs_)
        
        self._post_process(ax=self.ax, title=title, xlabel=xlabel, ylabel=ylabel, close=ax is None)
        return self


class Point(Graph):
    DEFAULT_FILENAME = 'point.png'

    def __init__(self, data:TDATA, x=None, y=None, group=None,
                 join=True, confidence=95, ax=None, name=None, savepath=None):
        super().__init__(data=data, kind=Graph.POINT, name=name, ax=ax, savepath=savepath)
        self.x = x
        self.y = y
        self.group = group
        self.size = None
        self.join = join
        self.confidence = confidence
        # setattr more

    def draw(self, ax=None, x=None, y=None, group=None, title=None, xlabel=None, ylabel=None, join=None, confidence=None, **kwargs):
        data, x, y, group, _ = self._check_inputs(self._data.dataframe, x, y, group, None)
        if join is None: join = self.join
        if confidence is None: confidence = self.confidence
        self.kwargs_ = kwargs.copy()
        self.kwargs_.update(data=data, hue=group, ax=ax, x=x, y=y, join=join, ci=confidence)

        self.ax = sns.pointplot(**self.kwargs_)
        
        self._post_process(ax=self.ax, title=title, xlabel=xlabel, ylabel=ylabel, close=ax is None)
        return self


class Heatmap(Graph):
    DEFAULT_FILENAME = 'heatmap.png'
    # default style
    DEFAULT_STYLE = dict(cmap='YlGnBu',
                        fmt='.1g',
                        font_size=8,    # annot_kws={'size': 8}
                        linewidths=0.5)

    def __init__(self, data:TDATA, annot:Union[bool, np.ndarray, pd.DataFrame]=None,
                name=None, ax=None, savepath=None):
        super().__init__(data=data, kind=Graph.HEATMAP, name=name, ax=ax, savepath=savepath)
        self.annot = annot

    def _convert_params(self, pairs:dict):
        if isinstance(pairs, dict) is False:
            return pairs

        if 'font_size' in pairs:
            if 'annot_kws' not in pairs:
                pairs['annot_kws'] = dict()
            pairs['annot_kws']['size'] = pairs.pop('font_size')

        if 'annot' in pairs and isinstance(pairs['annot'], bool) is False:
            pairs['fmt'] = ''

        return pairs

    def draw(self, ax=None, title=None, xlabel=None, ylabel=None, **kwargs):
        self.kwargs_ = self.DEFAULT_STYLE.copy()
        self.kwargs_.update(kwargs)
        self.kwargs_.update(data=self._data.dataframe, annot=self.annot, ax=ax)
        self.kwargs_ = self._convert_params(self.kwargs_)
        
        self.ax = sns.heatmap(**self.kwargs_)

        self._post_process(ax=self.ax, title=title, xlabel=xlabel, ylabel=ylabel, close=ax is None)
        return self


# TODO: define features like canvas
class Facet(Graph):
    TITLE_SIZE = 'large'
    TITLE_WEIGHT = 'bold'
    DEFAULT_FILENAME = 'facet.png'

    def set_title(self, _, title:str=None):
        if title is None and self.name is None:
            return
        
        title = title.upper() if title else self.name.upper()
        plt.title(title, fontdict={'fontsize':self.TITLE_SIZE, 'fontweight':self.TITLE_WEIGHT})
        # self.fig.set_titles(title, fontsize=self.TITLE_SIZE, fontweight=self.TITLE_WEIGHT)

    def set_labels(self, fig, xaxis=None, yaxis=None):
        if xaxis:
            fig.set_xlabels(xaxis, fontsize=self.TITLE_SIZE, fontweight=self.TITLE_WEIGHT)
        if yaxis:
            fig.set_ylabels(yaxis, fontsize=self.TITLE_SIZE, fontweight=self.TITLE_WEIGHT)

    def _post_process(self, fig=None, ax=None, title=None, xlabel=None, ylabel=None, close=True):
        self.set_title(plt, title)
        self.set_labels(fig, xlabel, ylabel)
        fig.tight_layout()

        if close:
            plt.close()


class RelPlot(Facet):
    DEFAULT_FILENAME = 'relplot.png'
    def __init__(self, data:TDATA, x=None, y=None,
                 group=None, kind=Graph.SCATTER, size=None, name=None, savepath=None):
        super().__init__(data=data, kind=kind, name=name, savepath=savepath)
        self.x = x
        self.y = y
        self.group = group
        self.size = size

    def draw(self, x=None, y=None, group=None, kind=None, size=None, title=None, xlabel=None, ylabel=None, **kwargs):
        if kind is None: kind = self.kind
        data, x, y, group, size = self._check_inputs(self._data.dataframe, x, y, group, size)
        self.kwargs_ = kwargs.copy()
        self.kwargs_.update(data=data, x=x, y=y, hue=group, kind=kind, size=size)

        self.fig = sns.relplot(**self.kwargs_)

        self._post_process(fig=self.fig, title=title, xlabel=xlabel, ylabel=ylabel, close=False)
        return self
