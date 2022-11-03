# BoFire
Bayesian Optimization Framework Intended for Real Experiments


## Dependencies

For development, we recommend to use [venv](https://docs.python.org/3/library/venv.html) as follows:
- Linux
  ```bash
  # init
  python -m venv .venv
  # activate
  source .venv/bin/activate
  # install dependencies
  pip install -r requirements.txt
  ```
- Windows
  ```bat
  .venv/Scripts/activate.bat
  pip install -r requirements.txt
  ```
## Tests

To run unit tests, execute the following:
- Linux
  ```bash
  # activate venv
  source .venv/bin/activate
  # add src to PYTHONPATH
  export PYTHONPATH=$(pwd)/src:$PYTHONPATH
  # run tests
  pytest tests/
  ```

- Windows 
  ```bat
  .venv/Scripts/activate.bat
  set PYTHONPATH=%cd%/src;%PYTHONPATH%
  pytest tests/
  ```

  