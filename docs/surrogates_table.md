**Surrogate**|**When to use**|**Type**
:-----:|:-----:|:-----:
[SingleTaskGPSurrogate](https://github.com/experimental-design/bofire/blob/main/bofire/surrogates/single\_task\_gp.py)|Optimizing a single objective with real valued inputs at a time under uncertainty|Gaussian process
[RandomForestSurrogate](https://github.com/experimental-design/bofire/blob/main/bofire/surrogates/random\_forest.py)|Optimizing a single objective at a time|sklearn random forest implementation 
[MLP](https://github.com/experimental-design/bofire/blob/main/bofire/surrogates/mlp.py)|Optimizing a single objective at a time|multi layer perceptron
[MixedSingleTaskGPSurrogate](https://github.com/experimental-design/bofire/blob/main/bofire/surrogates/mixed\_single\_task\_gp.py)|Optimizing a single objective with categorical and real valued inputs under uncertainty| Gaussian process