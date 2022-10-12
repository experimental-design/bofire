from typing import List, Tuple

import pandas as pd

from everest.domain.domain import Domain
from everest.strategies.strategy import Strategy


def propose_candidates(
    strategy: Strategy,
    domain: Domain,
    experiments: pd.DataFrame,
    candidate_count: int = 0,
    allow_insufficient_experiments: bool = None
) -> Tuple[pd.DataFrame, List[dict]]:
    strategy.init_domain(
        domain=domain,
    )
    strategy.tell(
        experiments=experiments,
    )
    candidates, configs = strategy.ask(
        candidate_count=candidate_count,
        allow_insufficient_experiments=allow_insufficient_experiments,
    )
    return candidates, configs
