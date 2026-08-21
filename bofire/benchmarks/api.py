from typing import Union

from bofire.benchmarks.benchmark import (
    Benchmark,
    FormulationWrapper,
    GenericBenchmark,
    SpuriousFeaturesWrapper,
    SyntheticBoTorch,
)
from bofire.benchmarks.detergent import Detergent
from bofire.benchmarks.hyperopt import Hyperopt
from bofire.benchmarks.multi import (
    BNH,
    C2DTLZ2,
    DTLZ2,
    TNK,
    ZDT1,
    CrossCoupling,
    MOMFBraninCurrin,
    SnarBenchmark,
)
from bofire.benchmarks.single import (
    Ackley,
    Booth,
    Branin,
    Branin30,
    CrossInTray,
    Easom,
    Hartmann,
    Hartmann6plus,
    Himmelblau,
    HolderTable,
    Multinormalpdfs,
    MultiTaskHimmelblau,
    Rosenbrock,
    SixHumpCamel,
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
    MOMFBraninCurrin,
]
AnySingleBenchmark = Union[
    Ackley,
    Booth,
    Branin,
    Branin30,
    CrossInTray,
    Easom,
    Hartmann,
    Hartmann6plus,
    Himmelblau,
    HolderTable,
    MultiTaskHimmelblau,
    Multinormalpdfs,
    Rosenbrock,
    SixHumpCamel,
]
