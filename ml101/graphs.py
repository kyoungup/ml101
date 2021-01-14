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
from sklearn.metrics import confusion_matrix
from ml101.config import ENV
import ml101.utils as utils
from ml101.evaluation import normalize_matrix


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

    def __init__(self, data: pd.DataFrame=None, type=None, name=None, ax=None, savepath=None):
        # TODO: Support Data class
        self.data = data
        self.type = type
        self.parent = None
        
        self.name = name
        self.fig = None
        self.axes = None
        self.ax = ax
        self.savepath = None
        self.filename = None
        if savepath:
            savepath = Path(savepath)
            if savepath.suffix != '':   # check if it is a file
                self.savepath = savepath.parent
                self.filename = savepath.name
            else:
                self.savepath = savepath
                self.filename = utils.insert2filename(self.DEFAULT_FILENAME, prefix=f'{utils.convert2filename(self.name)}_')
        else:
            self.filename = utils.insert2filename(self.DEFAULT_FILENAME, prefix=f'{utils.convert2filename(self.name)}_')

    @abstractmethod
    def draw(self, **kwargs):
        pass

    def show(self):
        plt.show()
        return self

    def set_title(self, title=None):
        if title is None and self.name is None:
            return
        
        title = title.upper() if title else self.name.upper()
        self.ax.set_title(title, fontsize=self.TITLE_SIZE, fontweight=self.TITLE_WEIGHT)

    def set_labels(self, xaxis=None, yaxis=None):
        if xaxis:
            self.ax.set_xlabel(xaxis, fontsize=self.TITLE_SIZE, fontweight=self.TITLE_WEIGHT)
        if yaxis:
            self.ax.set_ylabel(yaxis, fontsize=self.TITLE_SIZE, fontweight=self.TITLE_WEIGHT)

    def save(self, savepath=None):
        if savepath:
            savepath = Path(savepath)
            if savepath.suffix != '':   # check if it is a file
                filepath = savepath
            else:
                filepath = savepath / self.filename
        elif self.savepath:
            filepath = self.savepath / self.filename
        else:
            raise ValueError('No savepath is given.')

        if self.ax:
            self.ax.get_figure().savefig(filepath)
        elif self.fig:
            self.fig.savefig(filepath)
        return filepath
        

class Canvas(Graph):
    TITLE_SIZE = 'large'
    TITLE_WEIGHT = 'bold'
    DEFAULT_FILENAME = 'canvas.png'

    def __init__(self, name=None, data=None, savepath=None):
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
        if 'type' in kwargs:
            type = kwargs['type']
            if 'data' not in kwargs: kwargs['data'] = self.data
            if type == Graph.SCATTER:
                return Scatter(**kwargs)
            elif type == Graph.COUNT:
                return Count(**kwargs)
            elif type == Graph.LINE:
                return Line(**kwargs)
            else:
                raise ValueError(f'Unsupported Graph Type {type}')
        else:
            # TODO: automatically determin types and assign them
            raise ValueError('Graph type is missing')

    def set_title(self, title=None):
        if title is None and self.name is None:
            return
        
        title = title.upper() if title else self.name.upper()
        # TODO: set y position
        self.fig.suptitle(title, fontsize=self.TITLE_SIZE, fontweight=self.TITLE_WEIGHT)

    def draw(self, col_wrap:int=None, sharex=False, sharey=False, title=None, figsize:tuple=None):
        # TODO: col_wrap can be None
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
            graph.draw(ax=self.axes.flat[idx])

        self.set_title(title)
        self.fig.tight_layout()

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

    def __init__(self, data, x, y, group=None, size=None, ax=None, name=None, savepath=None, **kwargs):
        super().__init__(data=data, type=Graph.SCATTER, name=name, ax=ax, savepath=savepath)
        self.x = x
        self.y = y
        self.group = group
        self.size = size
        # setattr more

    def draw(self, ax=None, title=None, **kwargs):
        self.ax = ax
        self.kwargs = dict()
        # if 'ax' in kwargs: self.kwargs['ax'] = kwargs['ax']
        self.ax = sns.scatterplot(data=self.data, x=self.x, y=self.y, hue=self.group,
                                    size=self.size,
                                    ax=self.ax, **self.kwargs)
        self.set_title(title)

        if ax is None:
            self.ax.get_figure().tight_layout()
            plt.close()
        return self

class Line(Graph):
    DEFAULT_FILENAME = 'line.png'

    def __init__(self, data, x, y, group=None, size=None, ax=None, name=None, savepath=None, **kwargs):
        super().__init__(data=data, type=Graph.LINE, name=name, ax=ax, savepath=savepath)
        self.x = x
        self.y = y
        self.group = group
        self.size = size
        # setattr more

    def draw(self, ax=None, title=None, **kwargs):
        self.ax = ax
        self.kwargs = dict()
        # if 'ax' in kwargs: self.kwargs['ax'] = kwargs['ax']
        self.ax = sns.lineplot(data=self.data, x=self.x, y=self.y, hue=self.group,
                                    size=self.size,
                                    ax=self.ax, **self.kwargs)
        self.set_title(title)
        
        if ax is None:
            self.ax.get_figure().tight_layout()
            plt.close()
        return self


class Count(Graph):
    DEFAULT_FILENAME = 'count.png'

    def __init__(self, data, x, dir='vertical', group=None, ax=None, name=None, savepath=None, **kwargs):
        super().__init__(data=data, type=Graph.COUNT, name=name, ax=ax, savepath=savepath)
        self.x = x
        self.dir = dir
        self.group = group
        # setattr more

    def draw(self, ax=None, title=None, **kwargs):
        self.ax = ax
        self.kwargs = dict()
        # if 'ax' in kwargs: self.kwargs['ax'] = kwargs['ax']
        if self.dir == 'vertical':
            self.kwargs['x'] = self.x
        else:
            self.kwargs['y'] = self.x
        self.ax = sns.countplot(data=self.data, hue=self.group,
                                    ax=self.ax, **self.kwargs)
        self.set_title(title)
        
        if ax is None:
            self.ax.get_figure().tight_layout()
            plt.close()
        return self


class Heatmap(Graph):
    DEFAULT_FILENAME = 'heatmap.png'
    # default style
    DEFAULT_STYLE = dict(cmap='YlGnBu',
                        fmt='.1g',
                        font_size=8,    # annot_kws={'size': 8}
                        linewidths=0.5)

    def __init__(self, data, annot:Union[bool, np.ndarray, pd.DataFrame]=None,
                name=None, ax=None, savepath=None, **kwargs):
        super().__init__(data=data, type=Graph.HEATMAP, name=name, ax=ax, savepath=savepath)
        self.annot = annot

    def _convert_params(self, pairs:dict):
        if isinstance(pairs, dict) is False:
            return pairs

        if 'font_size' in pairs:
            if 'annot_kws' not in pairs:
                pairs['annot_kws'] = dict()
            pairs['annot_kws']['size'] = pairs['font_size']
            del pairs['font_size']

        return pairs

    def draw(self, ax=None, title=None, xlabel=None, ylabel=None, **kwargs):
        self.ax = ax

        self.kwargs = utils.update_kwargs(self.DEFAULT_STYLE, kwargs)
        self.kwargs = self._convert_params(self.kwargs)
        if self.annot is not None and isinstance(self.annot, bool) is False:
            self.kwargs['fmt'] = ''
        
        self.ax = sns.heatmap(data=self.data, annot=self.annot,
                              ax=self.ax, **self.kwargs)
        self.set_title(title)
        self.set_labels(xlabel, ylabel)

        if ax is None:
            self.ax.get_figure().tight_layout()
            plt.close()
        return self


class Facet(Graph):
    TITLE_SIZE = 'large'
    TITLE_WEIGHT = 'bold'
    DEFAULT_FILENAME = 'fecet.png'

    def set_title(self, title=None):
        if title is None and self.name is None:
            return
        
        title = title.upper() if title else self.name.upper()
        plt.title(title, fontdict={'fontsize':self.TITLE_SIZE, 'fontweight':self.TITLE_WEIGHT})
        # self.fig.set_titles(title, fontsize=self.TITLE_SIZE, fontweight=self.TITLE_WEIGHT)


class RelPlot(Facet):
    DEFAULT_FILENAME = 'relplot.png'
    def __init__(self, data:pd.DataFrame, type=Graph.SCATTER, name=None, savepath=None):
        super().__init__(data=data, type=Graph.SCATTER, name=name, savepath=savepath)

    def draw(self, x, y, group=None, type=None, sep_col=None, size=None):
        if type is None: type = self.type
        self.fig = sns.relplot(data=self.data, x=x, y=y, hue=group, kind=type, size=size)
        self.set_title()
        return self


class ConfusionMatrixGraph(Heatmap):
    DEFAULT_FILENAME = 'confusion_matrix.png'
    NORMALIZE_ALL = 'all'
    NORMALIZE_TRUE = 'true'
    NORMALIZE_PRED = 'pred'

    def __init__(self, cm, name=None, idx2label=None, savepath=None):
        self.idx2label = idx2label
        if idx2label:
            labels = list(idx2label.values())
            cm = pd.DataFrame(cm, index=labels, columns=labels)
        super().__init__(data=cm, name=name, savepath=savepath)
        
    def draw(self, suffix=None, normalize=NORMALIZE_TRUE, **kwargs):
        # Set title of figure
        title = f'Confusion Matrix{suffix}' if suffix else 'Confusion Matrix'

        cm = self.data
        # normalize
        cm_normalized, cm_sum = normalize_matrix(cm, normalize)

        annot = np.empty_like(cm).astype(str)
        for row, col in itertools.product(range(cm_normalized.shape[0]), range(cm_normalized.shape[1])):
            value = cm[row, col]
            normalized_value = cm_normalized[row, col]
            annot[row, col] = f'{normalized_value:.2%}\n{value}'
            if row == col:
                annot[row, col] += f'/{cm_sum[row, col]}'

        self.annot = annot
        self.data = cm_normalized
        super().draw(title=title, xlabel='Predicted labels', ylabel='True labels', **kwargs)
        self.data = cm
        return self

    
class Lines(Line):
    DEFAULT_FILENAME = 'line.png'

    def __init__(self, data, x, y, group=None, size=None, ax=None, name=None, savepath=None, **kwargs):
        super().__init__(data=data, type=Graph.LINE, name=name, ax=ax, savepath=savepath)
        self.x = x
        self.y = y
        self.group = group
        self.size = size
        # setattr more

    def draw(self, ax=None, title=None, **kwargs):
        self.ax = ax
        self.kwargs = dict()
        # if 'ax' in kwargs: self.kwargs['ax'] = kwargs['ax']
        self.ax = sns.lineplot(data=self.data, x=self.x, y=self.y, hue=self.group,
                                    size=self.size,
                                    ax=self.ax, **self.kwargs)
        self.set_title(title)
        
        if ax is None:
            plt.close()
        return self