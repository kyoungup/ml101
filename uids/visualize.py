from abc import ABCMeta, abstractmethod

from numpy.lib.npyio import save
from uids.config import ENV
import seaborn as sns
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import math
from typing import Union
import uids.utils as utils

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
    # Title Attr
    TITLE_SIZE = 'medium'
    TITLE_WEIGHT = 'normal'

    DEFAUL_FILENAME = 'graph.png'

    def __init__(self, data: pd.DataFrame=None, type=None, name=None, ax=None, savepath=None):
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
                self.filename = utils.insert2filename(self.DEFAUL_FILENAME, f'_{utils.convert2filename(self.name)}')
        else:
            self.filename = utils.insert2filename(self.DEFAUL_FILENAME, f'_{utils.convert2filename(self.name)}')

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
    DEFAUL_FILENAME = 'canvas.png'

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
    
    def remove(self, indices:list):
        for idx in indices:
            del self._graphs[idx]
            del self.axes[idx]

    def convert(self, **kwargs):
        if 'type' in kwargs:
            type = kwargs['type']
            if 'data' not in kwargs: kwargs['data'] = self.data
            if type == Graph.SCATTER:
                return Scatter(**kwargs)
            else:
                raise ValueError(f'Unsupported Graph Type {type}')
        else:
            # TODO: automatically determin types and assign them
            raise ValueError('Graph type is missing')

    def set_title(self, title=None):
        if title is None and self.name is None:
            return
        
        title = title.upper() if title else self.name.upper()
        self.fig.suptitle(title, fontsize=self.TITLE_SIZE, fontweight=self.TITLE_WEIGHT)

    def draw(self, col_wrap, sharex=False, sharey=False):
        num_graphs = len(self._graphs)
        num_cols = col_wrap if col_wrap < num_graphs else num_graphs
        num_rows = math.ceil(num_graphs / num_cols)
        self.fig, self.axes = plt.subplots(ncols=num_cols, nrows=num_rows,
                                           sharex=sharex, sharey=sharey,
                                           squeeze=False)
        num_axes = num_cols * num_rows                                           
        if num_graphs != num_axes:
            for idx in range(num_graphs, num_axes):
                self.fig.delaxes(self.axes.flat[idx])

        for idx, graph in enumerate(self._graphs):
            graph.draw(ax=self.axes.flat[idx])

        self.set_title()
        self.fig.tight_layout()

        return self

    def save(self, savepath=None, each=False):
        filepath = super().save(savepath)

        if each:
            for idx, graph in enumerate(self._graphs):
                isavepath = savepath.parent / utils.insert2filename(savepath.name, f'_{idx}')
                graph.save(isavepath)
        
        return filepath
        

class Scatter(Graph):
    DEFAUL_FILENAME = 'scatter.png'

    def __init__(self, data, x, y, group=None, size=None, ax=None, **kwargs):
        super().__init__(data=data, type=Graph.SCATTER, name=kwargs['name'], ax=ax)
        self.x = x
        self.y = y
        self.group = group
        self.size = size
        # setattr more

    def draw(self, ax=None, **kwargs):
        self.ax = ax
        self.kwargs = dict()
        # if 'ax' in kwargs: self.kwargs['ax'] = kwargs['ax']
        self.ax = sns.scatterplot(data=self.data, x=self.x, y=self.y, hue=self.group,
                                    size=self.size,
                                    ax=self.ax, **self.kwargs)
        self.set_title()
        
        if ax is None:
            plt.close()
        return self


class Fecet(Graph):
    TITLE_SIZE = 'large'
    TITLE_WEIGHT = 'bold'
    DEFAUL_FILENAME = 'fecet.png'

    def set_title(self, title=None):
        if title is None and self.name is None:
            return
        
        title = title.upper() if title else self.name.upper()
        plt.title(title, fontdict={'fontsize':self.TITLE_SIZE, 'fontweight':self.TITLE_WEIGHT})
        # self.fig.set_titles(title, fontsize=self.TITLE_SIZE, fontweight=self.TITLE_WEIGHT)


class RelPlot(Fecet):
    DEFAUL_FILENAME = 'relplot.png'
    def __init__(self, data:pd.DataFrame, type=Graph.SCATTER, name=None, savepath=None):
        super().__init__(data=data, type=Graph.SCATTER, name=name, savepath=savepath)

    def draw(self, x, y, group=None, type=None, sep_col=None, size=None):
        if type is None: type = self.type
        self.fig = sns.relplot(data=self.data, x=x, y=y, hue=group, kind=type, size=size)
        self.set_title()
        return self