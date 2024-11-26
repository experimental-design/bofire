import os
import re
import sys
from tempfile import gettempdir
from uuid import uuid4


def test_docs():
    """This tests tests the codeblocks embedded within the documentation"""
    folder_under_test = "docs"
    files_list = [
        os.path.join(folder_under_test, x)
        for x in os.listdir(folder_under_test)
        if x.rsplit(".")[-1] == "md"
    ]
    files_under_test = files_list + ["README.md"]
    code_block_start = "```python"
    code_block_end = "```"
    code_block_regex = re.compile(
        f"{code_block_start}(.*?){code_block_end}",
        flags=re.DOTALL,
    )
    for filename in files_under_test:
        print(f"### Processing doc file {filename} ###")
        with open(filename) as f:
            content = f.read()
        codeblocks = code_block_regex.findall(content)
        codeblocks = "\n".join(codeblocks)
        tmpfile = f"{gettempdir()}/{uuid4().hex}.py"
        exit_code = None
        try:
            with open(tmpfile, "w") as f:
                f.write(codeblocks)
            exit_code = os.system(f"{sys.executable} {tmpfile}")
        finally:
            os.remove(tmpfile)
        assert exit_code == 0
