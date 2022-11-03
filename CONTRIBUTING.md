
## Development Enviromnent

### Virtual Environments

For development of BoFire, we recommend to use a virtual environment, e.g., [venv](https://docs.python.org/3/library/venv.html) as follows:
- Linux
  ```bash
  # init
  python -m venv .venv
  # activate
  source .venv/bin/activate
  ```
- Windows
  ```bat
  .venv/Scripts/activate.bat
  ```
### Dependencies and Tests
You have at least two options to install dependencies and run unit tests. In both cases, you need to activate your environment first.
- Linux
  ```bash
  source .venv/bin/activate
  ```
- Windows 
  ```bat
  .venv/Scripts/activate.bat
  ```

#### Option 1 - `setup.py`

Using the `setup.py` you can install all dependencies and BoFire in editable mode via

```
pip install -e .[testing]
```
Afterwards, you can check that the tests are successful vie
```
pytest tests/
```
#### Option 2 - `requirements.txt`

Install the dependencies via
```
pip install -r requirements.txt
```
In contrast to Option 1 Since BoFire you have not installed BoFire. Hence, you have to add it to your `PYTHONPATH` environment variable such that Python can find BoFire.
- Linux
  ```bash
  export PYTHONPATH=$(pwd):$PYTHONPATH
  ```
- Windows
  ```bat
  set PYTHONPATH=%cd%;%PYTHONPATH%
  ```
Run unit tests via
```
pytest tests/
```
