from typing import Union

from bofire.benchmarks.aspen_benchmark import Aspen_benchmark
from bofire.benchmarks.benchmark import Benchmark, GenericBenchmark
from bofire.benchmarks.hyperopt import Hyperopt
from bofire.benchmarks.multi import C2DTLZ2, DTLZ2, ZDT1, CrossCoupling, SnarBenchmark
from bofire.benchmarks.single import Ackley, Branin, Branin30, Hartmann, Himmelblau

AnyMultiBenchmark = Union[C2DTLZ2, DTLZ2, ZDT1, CrossCoupling, SnarBenchmark]
AnySingleBenchmark = Union[Ackley, Branin, Branin30, Hartmann, Himmelblau]
