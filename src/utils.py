""" 
ML utility functions

author: Kai Chang, 
email: kaichang@mit.edu
GitHub: kai-ovo

"""

import numpy as np
import random, os
import torch 
from timeit import default_timer as timer

__all__ = ['global_seed', 
           'create_path', 
           'time_func', 
           'device',
           'plot_setting',
           'HyperParams']

def global_seed(seed: int):
    """
    Set seed
    """
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def create_path(path):
    """
    if the directory does not exist, then create it
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created.")
    else:
        print(f"Directory '{path}' already exists.")

def time_func(func, *args, **kwargs):
    """Calls func with given args and returns a (seconds, res) tuple.

    >>> def foo(a, b):
    ...    return a + b
    >>> t, res = time_func(foo, 1, b=2)
    >>> res
    3
    >>> isinstance(t, float)
    True
    >>> t < 1.
    True

    :param func: function to be evaluated
    :param args: args for func
    :param kwds: kwds for func
    :return: a tuple: (seconds, func(*args, **kwds)
    """
    tic = timer()
    res = func(*args, **kwargs)
    toc = timer()
    return toc - tic, res

def device(gpu=0):
    return f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'

def plot_setting():
    """
    Plotting settings
    """
    from matplotlib import rc
    import seaborn as sns
    sns.set()
    rc('font',**{'family':'sans-serif',
                 'sans-serif':['Helvetica'],
                 'size' : 16})
    rc('text', usetex=True)
    rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')


class HyperParams(object):

    """
    A class for saving hyperparameters

    Example:
    >>> params = {'k1' : val1, 
                  'k2' : val2}
    >>> hp = HyperParams(**params)
    >>> hp.k1
    val1
    >>> hp.k2
    val2

    """

    def __init__(self, **params):
        self.__dict__.update(params) 

    def update(self, **params):
        """
        update attributes (in this case, hyperparameters)
        """
        for key, value in params.items():
            setattr(self, key, value)