# this is based on the mlflow implementation: https://github.com/mlflow/mlflow/blob/master/mlflow/pytorch/pickle_module.py
import warnings


try:
    from pickle import Unpickler  # noqa: F401

    from cloudpickle import *  # noqa: F403  # type: ignore
    from cloudpickle import CloudPickler as Pickler  # noqa: F401

except ModuleNotFoundError:
    warnings.warn("Cloudpickle is not available.")
