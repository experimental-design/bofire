from typing import Union

from bofire.benchmarks.aspen_benchmark import Aspen_benchmark
from bofire.benchmarks.benchmark import Benchmark, GenericBenchmark
from bofire.benchmarks.detergent import Detergent
from bofire.benchmarks.hyperopt import Hyperopt
from bofire.benchmarks.multi import (
    BNH,
    C2DTLZ2,
    DTLZ2,
    TNK,
    ZDT1,
    CrossCoupling,
    SnarBenchmark,
)
from bofire.benchmarks.single import (
    Ackley,
    Branin,
    Branin30,
    Hartmann,
    Hartmann6plus,
    Himmelblau,
    Multinormalpdfs,
    MultiTaskHimmelblau,
)


AnyMultiBenchmark = Union[
    C2DTLZ2,
    Detergent,
    DTLZ2,
    ZDT1,
    CrossCoupling,
    SnarBenchmark,
    BNH,
    TNK,
]
AnySingleBenchmark = Union[
    Ackley,
    Branin,
    Branin30,
    Hartmann,
    Hartmann6plus,
    Himmelblau,
    MultiTaskHimmelblau,
    Multinormalpdfs,
]
