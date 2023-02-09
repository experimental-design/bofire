# this is copied from mlflow: https://github.com/mlflow/mlflow/blob/master/mlflow/pytorch/pickle_module.py

from pickle import Unpickler  # noqa: F401

from cloudpickle import CloudPickler as Pickler  # noqa: F401
from cloudpickle import *  # noqa: F401, F403
