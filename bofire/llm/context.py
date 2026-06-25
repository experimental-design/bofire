"""Runtime context passed to the LLM agent and its capability tools.

This is the dependency object (pydantic-ai ``deps``) injected into every
agent run. It carries only the strategy's intrinsic state — the domain, the
completed experiments, and the pending candidates — which capability tools
read via ``RunContext[LLMContext]``. Anything a specific capability needs that
is not intrinsic strategy state (e.g. a surrogate, a vector store) is carried
by that capability's own configuration, not here.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from bofire.data_models.domain.api import Domain


@dataclass
class LLMContext:
    domain: Domain
    experiments: Optional[pd.DataFrame]
    candidates: Optional[pd.DataFrame]
