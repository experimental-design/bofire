"""Registration utilities for custom LLM provider types."""

from bofire.data_models.unions import extract_union_args, tagged_union


def register_llm_provider(data_model_cls: type, overwrite: bool = False) -> None:
    """Register a custom LLM provider so it is accepted in AnyLLMProvider fields.

    Rebuilds the ``AnyLLMProvider`` union with the new type appended, and
    calls ``model_rebuild`` on dependent Pydantic models (``LLMStrategy``)
    so that the new type is accepted.

    Users must separately register a mapper function via
    ``bofire.llm.mapper.register`` so that the data model can be turned
    into a pydantic-ai ``Model`` at runtime.

    Args:
        data_model_cls: A concrete subclass of ``LLMProvider``.
        overwrite: If ``True``, replace an existing provider registered under
            the same ``type`` discriminator instead of raising.

    Raises:
        ValueError: If a different provider with the same ``type`` is already
            registered and *overwrite* is ``False``.
    """
    import bofire.data_models.llm.api as llm_api
    from bofire.data_models._register_utils import patch_field, register_into
    from bofire.data_models.strategies.llm import LLMStrategy

    # AnyLLMProvider has no backing registry list, so work on a list built from
    # the current union members and rebuild the union from it afterwards.
    existing_types, _ = extract_union_args(llm_api.AnyLLMProvider)
    types = list(existing_types)
    action, _ = register_into(
        types, data_model_cls, overwrite=overwrite, kind="LLM provider"
    )
    if action == "noop":
        return
    llm_api.AnyLLMProvider = tagged_union(*types)

    patch_field(LLMStrategy, "llm", llm_api.AnyLLMProvider)
    LLMStrategy.model_rebuild(force=True)
