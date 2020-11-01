import pandas as pd

class TableData(pd.DataFrame):
    def __init__(self):
        super().__init__()
    

class ListData(list):
    def __init__(self):
        super().__init__()


class Work:
    def __init__(self, name, obj):
        self.name = name
        self.object = obj
    
    def __str__(self):
        super().__str__()

    @property
    def name(self):
        return self.name


class WorkFlow:
    def __init__(self):
        self.works = list()
        self.workflows = list()