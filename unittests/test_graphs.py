import unittest
import ml101.matplot_backend
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
from ml101.graphs import Graph, Canvas
from ml101.graphs import RelPlot, Scatter, Line, Count, Heatmap
from ml101.graphs import Histogram


CWD = Path(__file__).parent
TEMP = CWD / 'temp'
TEMP.mkdir(exist_ok=True, parents=True)


class TestCanvas(unittest.TestCase):
    def setUp(self):
        self.penguins = sns.load_dataset('penguins')
        self.tips = sns.load_dataset('tips')
        self.c1 = Canvas(name='test - tips', data=self.tips, savepath=TEMP)
        self.c2 = Canvas(name='test - penguins', data=self.penguins, savepath=TEMP)
        self.savefile = None
        self.savefiles = None

    def tearDown(self) -> None:
        if self.savefile and self.savefile.exists():
            self.savefile.unlink()
        if self.savefiles:
            for savefile in self.savefiles:
                if savefile.exists(): savefile.unlink()

    def test_canvas(self):
        ex1 = [dict(kind=Graph.SCATTER, name='total-tip', x='total_bill', y='tip', group='day'),
                dict(kind=Graph.SCATTER, name='tip-size', x='tip', y='size', group='day'),
                dict(kind=Graph.SCATTER, name='total-tip2', x='total_bill', y='tip', group='day')
            ]
        self.c1.add(ex1)
        self.c1.draw(col_wrap=4)
        # self.c1.show()
        self.savefile = self.c1.save()
        assert Path(self.savefile).exists()

    def test_add_canvas(self):
        ex2 = [dict(kind=Graph.SCATTER, name='total-tip3', x='bill_length_mm', y='flipper_length_mm', group='species'),
               dict(kind=Graph.SCATTER, name='tip-size3', x='body_mass_g', y='flipper_length_mm', group='sex'),
               dict(kind=Graph.SCATTER, name='tip-size3', x='body_mass_g', y='flipper_length_mm', group='island')
            ]
        self.c2.add(ex2)
        self.c1.add(self.c2)
        self.c1.draw(col_wrap=4)
        # self.c1.show()
        self.savefile = self.c1.save()
        assert Path(self.savefile).exists()
    
    def test_save_each(self):
        ex1 = [dict(kind=Graph.SCATTER, name='total-tip', x='total_bill', y='tip', group='day'),
                dict(kind=Graph.SCATTER, name='tip-size', x='tip', y='size', group='day'),
                dict(kind=Graph.SCATTER, name='total-tip2', x='total_bill', y='tip', group='day')
            ]
        self.c1.add(ex1)
        self.c1.draw(col_wrap=4)
        # self.c1.show()
        self.savefile, self.savefiles = self.c1.save(each=True)
        assert Path(self.savefile).exists()
        assert self.savefile and len(self.savefiles) == len(ex1)


class TestScatter(unittest.TestCase):
    def setUp(self):
        self.savefile = list()
        self.penguins = sns.load_dataset('penguins')    # ['species', 'island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']
        self.g = Scatter(name='penguins', data=self.penguins, x="flipper_length_mm", y="bill_length_mm", group="sex")

    def tearDown(self) -> None:
        for savefile in self.savefile:
            if savefile.exists():
                savefile.unlink()

    def test_draw(self):
        self.g.draw()
        # self.g.show()
        self.savefile.append( self.g.save(TEMP) )

    def test_draw_x_y_None(self):
        self.g.x = None
        self.g.y = None
        self.savefile.append( self.g.draw().save(TEMP) )

    def test_draw_x_None_y_list(self):
        self.g.x = None
        self.g.y = ['bill_length_mm', 'bill_depth_mm']
        self.savefile.append( self.g.draw().save(TEMP) )

    def test_draw_x_str_y_list(self):
        self.g.x = 'flipper_length_mm'
        self.g.y = ['bill_length_mm', 'bill_depth_mm']
        self.savefile.append( self.g.draw().save(TEMP) )

    def test_draw_y_None_x_list(self):
        self.g.y = None
        self.g.x = ['bill_length_mm', 'bill_depth_mm']
        self.savefile.append( self.g.draw().save(TEMP) )

    def test_draw_x_series_y_series(self):
        self.g.y = self.penguins['bill_depth_mm']
        self.g.x = self.penguins['body_mass_g']
        self.savefile.append( self.g.draw().save(TEMP) )

    def test_reuse(self):
        self.savefile.append( self.g.draw(x=self.penguins['body_mass_g'], y=self.penguins['bill_depth_mm']).save(TEMP / 'graph1.png') )
        self.savefile.append( self.g.draw(x=['bill_length_mm', 'bill_depth_mm']).save(TEMP / 'graph2.png') )


class TestLine(unittest.TestCase):
    def setUp(self):
        self.savefile = None
        self.flights = sns.load_dataset('flights')  # ['year', 'month', 'passengers']
        self.g = Line(name='flights', data=self.flights, x="year", y="passengers", group="month", size='month')
        self.gw = Line(name='flights', data=self.flights)

    def tearDown(self) -> None:
        if self.savefile and self.savefile.exists():
            self.savefile.unlink()

    def test_draw(self):
        self.g.draw()
        # self.g.show()
        self.savefile = self.g.save(TEMP)
        assert self.savefile.exists()

    def test_draw_x_y_None_wideform(self):
        self.savefile = self.gw.draw().save(TEMP)
        assert self.savefile.exists()

    def test_draw_x_None_y_list(self):
        self.savefile = self.g.draw(y = ['year', 'passengers']).save(TEMP)
        assert self.savefile.exists()

    def test_draw_y_None_x_list(self):
        self.savefile = self.g.draw(x = ['year', 'passengers']).save(TEMP)
        assert self.savefile.exists()

    def test_draw_x_series_y_series(self):
        self.savefile = self.g.draw(x = self.flights['month'], y = self.flights['passengers']).save(TEMP)
        assert self.savefile.exists()
    
    def test_draw_x_str_y_list(self):
        self.savefile = self.g.draw(x = 'year', y = ['passengers', 'month'], title='custom flights').save(TEMP)
        assert self.savefile.exists()


class TestCount(unittest.TestCase):
    def setUp(self):
        self.savefile = None
        self.data = sns.load_dataset('titanic')  # ['year', 'month', 'passengers']
        self.g = Count(name='titanic', data=self.data)

    def tearDown(self) -> None:
        if self.savefile and self.savefile.exists():
            self.savefile.unlink()

    def test_draw_vertical(self):
        self.g.draw(x="class", group="who")
        self.savefile = self.g.save(TEMP)
        assert self.savefile.exists()
    
    def test_draw_horizental(self):
        self.g.draw(y='class', group='who')
        self.savefile = self.g.save(TEMP)
        assert self.savefile.exists()


class TestHeatmap(unittest.TestCase):
    def setUp(self):
        self.savefile = None
        self.data = np.corrcoef(np.random.randn(10, 10))
        self.g = Heatmap(name='corr', data=self.data, annot=True)

    def tearDown(self) -> None:
        if self.savefile and self.savefile.exists():
            self.savefile.unlink()

    def test_graph_heatmap(self):
        self.g.draw(title='Random Correlation')
        self.savefile = self.g.save(TEMP)
        assert self.savefile.exists()





class TestRelplot(unittest.TestCase):
    def setUp(self):
        self.savefile = None
        self.tips = sns.load_dataset('tips')    # ['total_bill', 'tip', 'sex', 'smoker', 'day', 'time', 'size']
        self.g = RelPlot(name='tips', data=self.tips, savepath=TEMP)

        self.fmri = sns.load_dataset("fmri")    # ['subject', 'timepoint', 'event', 'region', 'signal']
        self.lg = RelPlot(name='fmri', data=self.fmri, savepath=TEMP)

    def tearDown(self) -> None:
        if self.savefile and self.savefile.exists():
            self.savefile.unlink()

    def test_graph_scatter(self):
        self.g.draw(x="total_bill", y="tip", group="smoker", size="size")
        self.savefile = self.g.save()
        assert self.savefile.exists()

    def test_graph_line(self):
        self.lg.draw(kind=Graph.LINE, x="timepoint", y="signal", group="region")
        self.savefile = self.lg.save()
        assert self.savefile.exists()

    def test_draw_x_y_None(self):
        self.savefile = self.g.draw(x=None, y=None, ylabel='tip', title='tips-all').save()
        assert self.savefile.exists()

    def test_draw_x_None_y_list(self):
        self.savefile = self.g.draw(x=None, y=['total_bill', 'tip']).save()
        assert self.savefile.exists()

    def test_draw_y_None_x_list(self):
        self.savefile = self.g.draw(x=['total_bill', 'tip'], y=None).save()
        assert self.savefile.exists()

    def test_draw_x_series_y_series(self):
        self.savefile = self.g.draw(x=self.tips['tip'], y=self.tips['total_bill']).save()
        assert self.savefile.exists()


class TestHistogram(unittest.TestCase):
    def setUp(self) -> None:
        self.savefile = None
        self.data = sns.load_dataset('penguins')      # ['species', 'island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']

    def tearDown(self) -> None:
        if self.savefile and self.savefile.exists():
            self.savefile.unlink()

    def test_draw(self):
        self.g = Histogram(name='penguins', data=self.data, x='flipper_length_mm')
        self.savefile = self.g.draw().save(TEMP)
        assert self.savefile.exists()

    def test_draw_cols(self):
        self.g = Histogram(name='penguins', data=self.data, x=['flipper_length_mm', 'body_mass_g'])
        self.savefile = self.g.draw().save(TEMP)
        assert self.savefile.exists()

    def test_draw_all(self):
        self.g = Histogram(name='penguins', data=self.data)
        self.savefile = self.g.draw(xlabel='All Variables').save(TEMP)
        assert self.savefile.exists()