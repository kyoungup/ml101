import unittest
import uids.matplot_backend
import pandas as pd
import seaborn as sns
from pathlib import Path

import uids.utils as utils
from uids.visualize import Graph, Canvas, RelPlot, Scatter

CWD = Path(__file__).parent
TEMP = CWD / 'temp'
TEMP.mkdir(exist_ok=True, parents=True)


class Test_Canvas(unittest.TestCase):
    def setUp(self):
        self.penguins = sns.load_dataset('penguins')
        self.tips = sns.load_dataset('tips')
        self.c1 = Canvas(name='test - tips', data=self.tips, savepath=TEMP)
        self.c2 = Canvas(name='test - penguins', data=self.penguins, savepath=TEMP)
        self.savefile = None

    def tearDown(self) -> None:
        if self.savefile and self.savefile.exists():
            self.savefile.unlink()
        pass

    def test_canvas(self):
        ex1 = [dict(type=Graph.SCATTER, name='total-tip', x='total_bill', y='tip', group='day'),
                dict(type=Graph.SCATTER, name='tip-size', x='tip', y='size', group='day'),
                dict(type=Graph.SCATTER, name='total-tip2', x='total_bill', y='tip', group='day')
            ]
        self.c1.add(ex1)
        self.c1.draw(col_wrap=4)
        # self.c1.show()
        self.savefile = self.c1.save()
        assert Path(self.savefile).exists()

    def test_add_canvas(self):
        ex2 = [dict(type=Graph.SCATTER, name='total-tip3', x='bill_length_mm', y='flipper_length_mm', group='species'),
               dict(type=Graph.SCATTER, name='tip-size3', x='body_mass_g', y='flipper_length_mm', group='sex'),
               dict(type=Graph.SCATTER, name='tip-size3', x='body_mass_g', y='flipper_length_mm', group='island')
            ]
        self.c2.add(ex2)
        self.c1.add(self.c2)
        self.c1.draw(col_wrap=4)
        # self.c1.show()
        self.savefile = self.c1.save()
        assert Path(self.savefile).exists()
    
    def test_save_each(self):
        ex1 = [dict(type=Graph.SCATTER, name='total-tip', x='total_bill', y='tip', group='day'),
                dict(type=Graph.SCATTER, name='tip-size', x='tip', y='size', group='day'),
                dict(type=Graph.SCATTER, name='total-tip2', x='total_bill', y='tip', group='day')
            ]
        self.c1.add(ex1)
        self.c1.draw(col_wrap=4)
        # self.c1.show()
        self.savefile, self.isavefile = self.c1.save(each=True)
        assert Path(self.savefile).exists()
        assert len(self.isavefile) == len(ex1)

class Test_SCatter(unittest.TestCase):
    def setUp(self):
        self.savefile = None
        self.penguins = sns.load_dataset('penguins')
        self.g = Scatter(name='penguins', data=self.penguins, x="flipper_length_mm", y="bill_length_mm", group="sex")

    def tearDown(self) -> None:
        if self.savefile and self.savefile.exists():
            self.savefile.unlink()
        pass

    def test_draw(self):
        self.g.draw()
        # self.g.show()
        self.savefile = self.g.save(TEMP)


class Test_Relplot(unittest.TestCase):
    def setUp(self):
        self.savefile = None
        self.tips = sns.load_dataset('tips')
        self.g = RelPlot(name='tips', data=self.tips, savepath=TEMP)

        self.fmri = sns.load_dataset("fmri")
        self.lg = RelPlot(name='fmri', data=self.fmri, savepath=TEMP)

    def tearDown(self) -> None:
        if self.savefile and self.savefile.exists():
            self.savefile.unlink()
        pass

    def test_graph_scatter(self):
        self.g.draw(x="total_bill", y="tip", group="smoker", size="size")
        self.savefile = self.g.save()

    def test_graph_line(self):
        self.lg.draw(type=Graph.LINE, x="timepoint", y="signal", group="region")
        self.savefile = self.lg.save()
