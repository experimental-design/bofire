import bofire.surrogates.api as surrogates
from bofire.benchmarks.single import MultiFidelityHimmelblau
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.surrogates.api import (
    MultiTaskGPSurrogate,
)

benchmark = MultiFidelityHimmelblau()
samples = benchmark.domain.inputs.sample(n=50)
experiments = benchmark.f(samples, return_complete=True)

# make fid the columns in order [fid, x_1, x_2, y, valid_y]
experiments = experiments[["task_id", "x_1", "x_2", "y", "valid_y"]]

input_features = benchmark.domain.inputs
output_features = benchmark.domain.outputs

# we setup the data model, here a Multi Task GP
surrogate_data = MultiTaskGPSurrogate(
    inputs=input_features,
    outputs=output_features,
    input_preprocessing_specs={"task_id": CategoricalEncodingEnum.ONE_HOT},
)

# we generate the json spec
# jspec = surrogate_data.json()

# surrogate_data = parse_obj_as(MultiTaskGPSurrogate, json.loads(jspec))
# surrogate_data = TypeAdapter(MultiTaskGPSurrogate).validate_python(json.loads(jspec))

surrogate = surrogates.map(surrogate_data)

surrogate.fit(experiments=experiments)

# dump it
# dump = surrogate.dumps()

# predict with it
df_predictions = surrogate.predict(experiments)
# transform to spec
predictions = surrogate.to_predictions(predictions=df_predictions)

# surrogate_data = parse_obj_as(AnySurrogate, json.loads(jspec))
# surrogate = surrogates.map(surrogate_data)
# surrogate.loads(dump)

# predict with it
# df_predictions2 = surrogate.predict(experiments)
# transform to spec
# predictions2 = surrogate.to_predictions(predictions=df_predictions2)

# assert predictions.equals(predictions2)
