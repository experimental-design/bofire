
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


You can install all dependencies and BoFire in editable mode via

```
pip install -e .[testing]
```
Afterwards, you can check that the tests are successful vie
```
pytest tests/
```
