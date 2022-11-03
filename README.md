# BoFire
Bayesian Optimization Framework Intended for Real Experiments


## Dependencies

For development, we recommend to use [venv](https://docs.python.org/3/library/venv.html) as follows:

```bash
# init
python -m venv .venv
# activate
source .venv/bin/activate
# install dependencies
pip install -r requirements.txt
```

When adding new dependencies, remember to update the [requirements.txt](./requirements.txt):

```bash
# install
pip install <pkg>
# update requirements.txt
pip freeze > requirements.txt
```

Any new non-dev dependency must also be added to [setup.py](./setup.py).



## Tests

To run unit tests, execute the following on Linx.

```bash
# activate venv
source .venv/bin/activate
# add src to PYTHONPATH
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
# run tests
pytest tests/
```

On Windows you can execute tests as follows.
 
```bat
.venv/Scripts/activate.bat
set PYTHONPATH=%cd%/src;%PYTHONPATH%
pytest tests/
```

