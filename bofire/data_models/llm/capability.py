from typing import Any, Literal

from pydantic import Field

from bofire.data_models.base import BaseModel


class LLMCapability(BaseModel):
    """Base class for all LLM capability configurations.

    A capability is a serializable, composable unit that extends what an LLM
    recommender (e.g. ``LLMStrategy``) can do. It bundles — depending on the
    concrete type — instructions, tools, model settings, and lifecycle hooks
    into a single object that is attached to the underlying agent.

    Like LLM providers, capabilities are pure configuration: the data model
    serializes the ``type`` discriminator plus typed config fields, while the
    functional mapper (``bofire.llm.capabilities_mapper``) reconstructs the
    live pydantic-ai capability — including any tool functions — at runtime.
    A capability's tools are therefore never serialized; they are rebuilt by
    the registered mapper keyed on ``type``.

    Attributes:
        type: Discriminator for the concrete capability type.
    """

    type: Any


class ExperimentAccessCapability(LLMCapability):
    """Exposes prior experiments and pending candidates to the LLM via tools.

    Instead of rendering experiments into the prompt as a fixed view, this
    capability gives the agent tool calls to inspect the experiment table on
    demand (recent rows, top rows by objective, nearest neighbours, summary
    statistics) and to list pending candidates. The capability's instructions
    surface only the *counts* of experiments and pending candidates, which
    motivates the model to call the tools rather than imposing a predefined
    view.

    Enabled by default on ``LLMStrategy``. Note that supplying an explicit
    ``capabilities`` list to the strategy replaces this default, so re-add it
    if you also pass other capabilities.

    Attributes:
        max_rows_per_tool_call: Upper bound on the number of experiment rows a
            single tool call may return, to keep tool outputs bounded.
    """

    type: Literal["ExperimentAccessCapability"] = "ExperimentAccessCapability"
    max_rows_per_tool_call: int = Field(
        default=50,
        gt=0,
        description="Maximum number of experiment rows returned per tool call.",
    )
