from typing import List


def get_invalids(valid: dict) -> List[dict]:
    return [
        {
            k: v for k, v in valid.items()
            if k != k_
        }
        for k_ in valid.keys()
    ]


INVALID_SPECS = [
    [1, 2, 3],
    {"a": "qwe"},
]
