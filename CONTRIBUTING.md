
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
  python -m venv .venv
  .venv/Scripts/activate.bat
  ```
  
### Dependencies and Tests
To install dependencies and run unit tests, activate your environment first.
- Linux
  ```bash
  source .venv/bin/activate
  ```
- Windows 
  ```bat
  .venv/Scripts/activate.bat
  ```


You can install all dependencies and BoFire in editable mode via

```
pip install -e .[testing]
```
Afterwards, you can check that the tests are successful via
```
pytest tests/
```

### Pre-commit Hooks
We use [Black](https://github.com/psf/black), [Flake8](https://flake8.pycqa.org/en/latest/), and [Isort](https://github.com/PyCQA/isort) as pre-commit hooks. Further, we use corresponding checks in the Github pipeline. To install the
pre-commit hooks defined in `.pre-commit-config.yaml`, you can proceed as follows.
```
pip install pre-commit
pre-commit install
```
