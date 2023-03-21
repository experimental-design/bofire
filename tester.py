import tests.bofire.data_models.specs.api as specs
from bofire.data_models.api import Domain, Inputs
from bofire.data_models.features.api import ContinuousInput

input_feature = ContinuousInput(key="x", bounds=(0, 10))
exit()
input_features = [input_feature]
inputs = Inputs(input_features=input_features)

domain = Domain(input_features=input_features)

myspecs = specs.features.valid(ContinuousInput).obj(bounds=(0, 1))
print(myspecs)
