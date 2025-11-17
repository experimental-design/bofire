
def make_surrogate(surrogate_type, data_model_type, locals_of_make: dict):
    """Factory function to create a surrogate of type surrogate_type from a data model of type data_model_type.
    This function is a helper for the `make`-`@classmethod`s of the surrogates. All locals that are not None are passed to the
    surrogate constructor. The ones that are None are not passed and hence their default values are used.
    Args:
        surrogate_type: The class of the surrogate to be created.
        data_model_type: The data model class.
        locals_of_make: The local variables of the make-function that called this function.
    Returns:
        Surrogate: The surrogate object.
    """
    locals_of_make = {k: v for k, v in locals_of_make.items() if v is not None}
    # since we get all locals from the `make`-`@classmethod`s we need to remove the `cls` variable.
    locals_of_make.pop("cls", None)
    return surrogate_type.from_spec(data_model_type(**locals_of_make))
