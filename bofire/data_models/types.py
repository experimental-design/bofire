from typing import Any


class NonExistingImportWrapper:
    def __init__(self, module_name: str):
        self._module_name = module_name

    def __getattr__(self, _: str) -> Any:
        raise ImportError(f"Module {self._module_name} does not exist.")

    def __call__(self, *_arg, **_kwargs) -> Any:
        raise ImportError(f"Module {self._module_name} does not exist.")
