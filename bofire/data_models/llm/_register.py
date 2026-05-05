"""Registration utilities for custom LLM provider types."""

from bofire.data_models.unions import extract_union_args, tagged_union


def register_llm_provider(data_model_cls: type) -> None:
    """Register a custom LLM provider so it is accepted in AnyLLMProvider fields.

    Rebuilds the ``AnyLLMProvider`` union with the new type appended, and
    calls ``model_rebuild`` on dependent Pydantic models (``LLMStrategy``)
    so that the new type is accepted.

    Users must separately register a mapper function via
    ``bofire.llm.mapper.register`` so that the data model can be turned
    into a pydantic-ai ``Model`` at runtime.

    Args:
        data_model_cls: A concrete subclass of ``LLMProvider``.
    """
    import bofire.data_models.llm.api as llm_api
    from bofire.data_models._register_utils import patch_field
    from bofire.data_models.strategies.llm import LLMStrategy

    existing_types, _ = extract_union_args(llm_api.AnyLLMProvider)
    if data_model_cls in existing_types:
        return
    llm_api.AnyLLMProvider = tagged_union(*existing_types, data_model_cls)

    patch_field(LLMStrategy, "llm", llm_api.AnyLLMProvider)
    LLMStrategy.model_rebuild(force=True)
