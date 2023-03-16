from bofire.surrogates import api as models


def test_surrogate_can_be_loaded_from_data_model(surrogate_spec):
    data_model = surrogate_spec.obj()
    model = models.map(data_model=data_model)
    assert model is not None
