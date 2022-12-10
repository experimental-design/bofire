from cgi import test
from functools import partial
from bofire.benchmarks.zdt import ZDT1
import bofire.benchmarks.benchmark as benchmark
from bofire.samplers import RejectionSampler
from bofire.strategies.botorch.sobo import BoTorchSoboStrategy, qEI


def test_benchmark():
    zdt1 = ZDT1(n_inputs=5)
    sobo_factory = partial(BoTorchSoboStrategy, acquisition_function=qEI())

    def sample(domain):
        n_initial_samples = 10
        sampler = RejectionSampler(domain=domain)
        sampled = sampler.ask(n_initial_samples)
        return sampled

    benchmark.run(
        zdt1,
        sobo_factory,
        n_iterations=5,
        metric=benchmark.best_multiplicative,
        initial_sampler=sample
    )


if __name__ == "__main__":
    test_benchmark()
