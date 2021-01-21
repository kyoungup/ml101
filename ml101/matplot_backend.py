try:
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'notebook')
except:
    import matplotlib
    if matplotlib.get_backend().lower() not in ['gtk3agg', 'gtk3cairo','macosx', 'nbagg',
                                        'qt4agg', 'qt4cairo', 'qt5agg', 'qt5cairo',
                                        'tkagg', 'tkcairo', 'webagg', 'wx', 'wxagg', 'wxcairo']:
        matplotlib.use('agg')
