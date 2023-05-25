import uuid
from contextlib import contextmanager
from pathlib import Path
from tempfile import gettempdir
from typing import Optional


@contextmanager
def make_tmpfile(name: Optional[str] = None):
    name = name or uuid.uuid4().hex
    test_folder = Path(gettempdir()) / "bofire"
    test_folder.mkdir(parents=True, exist_ok=True)
    test_file = test_folder / name
    try:
        yield test_file
    finally:
        test_file.unlink()
