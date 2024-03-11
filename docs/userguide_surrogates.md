# Surrogate models
In Bayesian Optimization, information from previous experiments is taken into account to generate proposals for future experiments. This information is leveraged by creating a surrogate model for the black-box function that is to be optimized based on the already available data. This surrogate model serves as an approximation to the black-box function and subsequent experimental proposals are based on it. Naturally, experimental candidates for which the surrogate model makes a promosing approximation of the output values should be chosen over ones for which this is not the case. However, since the available data might cover only a small part of the input space, it is likely to only be able to make uncertain approximations to values of the black-box function far away from the data. Therefore, the surrogate model should be able to express such a measure for uncertainty next to the approximated function value.
For the decision about which experiments to propose, one can additionally specify the acquisition function. This function determines if one rather wants to focus on exploitation, i.e., quickly approaching a close local optimum of the black-box function, or on exploration, i.e., exploring multiple regions of the input space first.
Therefore, three criteria typically determine the next experimental proposals: the value of the surrogate model, its uncertainty, and the acquisition function.

## Surrogate model options
BoFire offers the following classes of surrogate models.

**Surrogate**|**When to use**|**Type**
:-----:|:-----:|:-----:
[SingleTaskGPSurrogate](https://github.com/experimental-design/bofire/blob/main/bofire/surrogates/single\_task\_gp.py)|Optimizing a single objective with real valued inputs at a time under uncertainty|Gaussian process
[RandomForestSurrogate](https://github.com/experimental-design/bofire/blob/main/bofire/surrogates/random\_forest.py)|Optimizing a single objective at a time|sklearn random forest implementation 
[MLP](https://github.com/experimental-design/bofire/blob/main/bofire/surrogates/mlp.py)|Optimizing a single objective at a time|multi layer perceptron
[MixedSingleTaskGPSurrogate](https://github.com/experimental-design/bofire/blob/main/bofire/surrogates/mixed\_single\_task\_gp.py)|Optimizing a single objective with categorical and real valued inputs under uncertainty|Gaussian process
[XGBoostSurrogate](https://github.com/experimental-design/bofire/blob/main/bofire/surrogates/xgb.py)|Optimizing a single objective at a time|xgboost implementation of gradient boosting trees
[TanimotoGP](https://github.com/experimental-design/bofire/blob/main/bofire/surrogates/tanimoto_gp.py)|At least one input feature is a molecule represented as fingerprint|Gaussian process on a molecule space for which Tanimoto similarity determines the similarity between points


- The standard Kernel for all Gaussian Process (GP) strategies is a 5/2 matern kernel with automated relevance detection and normalization of the input features.
- The tree-based models (RandomForestSurrogate and XGBoostSurrogate) do not have Kernels but quantify uncertainty through a standard deviation of the predictions of their individual trees.
- MLP quantifies uncertainty be the standard deviation of multiple predictions that used different dropouts (randomly setting neural network weights to zero).

## Customization
BoFire also offers the option to customize surrogate models.

### Kernel customization
In particular, it is possible to customize the SingleTaskGPSurrogate in the following ways:
- Specify the Kernel. Options are defined in the [API file](https://github.com/experimental-design/bofire/blob/main/bofire/data_models/kernels/api.py).

**Kernel**|**When to use**|**Variable type**
:-----:|:-----:|:-----:
RBFKernel|Assuming closeness in points yields similartiy in the output value|[Continuous](https://github.com/experimental-design/bofire/blob/main/bofire/data_models/features/continuous.py)
MaternKernel|Same as RBFKernel with. Allows setting a smoothness parameter|Continuous
LinearKernel|Assumes linear dependence between output and input variables|Continuous
PolynomialKernel|Assumes polynomial dependence between output and input variables|Continuous
TanimotoKernel|Measures similarities between molecular inputs using Tanimoto Similiarity|[MolecularInput](https://github.com/experimental-design/bofire/blob/main/bofire/data_models/features/molecular.py)
HammondDistanceKernel|Similarity is defined by number of equal entries between two vectors (e.g., in One-Hot-encoding)|[Categorical](https://github.com/experimental-design/bofire/blob/main/bofire/data_models/features/categorical.py)


- SingleTaskGPSurrogate with PolynomialKernel is equivalent with PolynomialSurrogate. SingleTaskGPSurrogate with TanimotoKernel is equivalent with TanimotoGP.
- One can combine two Kernels by using AdditiveKernel or MultiplicativeKernel.

### Noise model customization

For experimental data being subject to noise, one can specify this noise. Options are:
**Noise Model**|**When to use**
:-----:|:-----:
[NormalPrior](https://github.com/experimental-design/bofire/blob/main/bofire/data_models/priors/normal.py)|Noise is Gaussian
[GammaPrior](https://github.com/experimental-design/bofire/blob/main/bofire/data_models/priors/gamma.py)|Noise has a Gamma distribution





