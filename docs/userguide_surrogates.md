# Surrogate models
In Bayesian Optimization, information from previous experiments is taken into account to generate proposals for future experiments. This information is leveraged by creating a surrogate model for the black-box function that is to be optimized based on the already available data. This surrogate model serves as an approximation to the black-box function and subsequent experimental proposals are based on it. Naturally, experimental candidates for which the surrogate model makes a promosing approximation of the output values should be chosen over ones for which this is not the case. However, since the available data might cover only a small part of the input space, the model is likely to only be able to make uncertain approximations to values of the black-box function far away from the data. Therefore, the surrogate model should be able to express such a measure for uncertainty next to the approximated function value.

For the decision about which experiments to propose, one can additionally specify the acquisition function. This function determines if one rather wants to focus on exploitation, i.e., quickly approaching a close local optimum of the black-box function, or on exploration, i.e., exploring different regions of the input space first.

Therefore, three criteria typically determine whether any candidate is selected as experimental proposal: the value of the surrogate model, the uncertainty of the model, and the acquisition function.

## Surrogate model options
BoFire offers the following classes of surrogate models.

**Surrogate**|**Optimization of**|**When to use**|**Type**
:-----:|:-----:|:-----:|:-----:
[SingleTaskGPSurrogate](https://github.com/experimental-design/bofire/blob/main/bofire/surrogates/single\_task\_gp.py)|a single objective with real valued inputs|Limited data and black-box function is smooth|Gaussian process
[RandomForestSurrogate](https://github.com/experimental-design/bofire/blob/main/bofire/surrogates/random\_forest.py)|a single objective|Rich data; black-box function does not have to be smooth|sklearn random forest implementation 
[MLP](https://github.com/experimental-design/bofire/blob/main/bofire/surrogates/mlp.py)|a single objective with real-valued inputs|Rich data and black-box function is smooth|Multi layer perceptron
[MixedSingleTaskGPSurrogate](https://github.com/experimental-design/bofire/blob/main/bofire/surrogates/mixed\_single\_task\_gp.py)|a single objective with categorical and real valued inputs|Limited data and black-box function is smooth|Gaussian process
[XGBoostSurrogate](https://github.com/experimental-design/bofire/blob/main/bofire/surrogates/xgb.py)|a single objective|Rich data; black-box function does not have to be smooth|xgboost implementation of gradient boosting trees
[TanimotoGP](https://github.com/experimental-design/bofire/blob/main/bofire/surrogates/tanimoto_gp.py)|a single objective|At least one input feature is a molecule represented as fingerprint|Gaussian process on a molecule space for which Tanimoto similarity determines the similarity between points|


- The standard Kernel for all Gaussian Process (GP) strategies is a 5/2 matern kernel with automated relevance detection and normalization of the input features.
- The tree-based models (RandomForestSurrogate and XGBoostSurrogate) do not have Kernels but quantify uncertainty through a standard deviation of the predictions of their individual trees.
- MLP quantifies uncertainty be the standard deviation of multiple predictions that come from different dropouts (randomly setting neural network weights to zero).

## Customization
BoFire also offers the option to customize surrogate models. In particular, it is possible to customize the SingleTaskGPSurrogate in the following ways.

### Kernel customization
Specify the [Kernel](https://github.com/experimental-design/bofire/blob/main/bofire/data_models/kernels/api.py):

**Kernel**|**Description**|**Input variable type**
:-----:|:-----:|:-----:
[RBFKernel](https://en.wikipedia.org/wiki/Radial_basis_function_kernel)|Based on Gaussian distribution; translation invariant and isotropic|[Continuous](https://github.com/experimental-design/bofire/blob/main/bofire/data_models/features/continuous.py)
[MaternKernel](https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function)|Based on Gamma function; translation invariant and isotropic; allows setting a smoothness parameter|Continuous
[PolynomialKernel](https://scikit-learn.org/stable/modules/metrics.html)|Based on dot-product of two vectors of input points; not translation invariant; rotation invariant|Continuous
[LinearKernel](https://scikit-learn.org/stable/modules/metrics.html)|Equal to dot-product of two vectors of input points; not translation invariant; rotation invariant|Continuous
TanimotoKernel|Measures similarities between molecular inputs using [Tanimoto Similiarity](https://en.wikipedia.org/wiki/Jaccard_index)|[MolecularInput](https://github.com/experimental-design/bofire/blob/main/bofire/data_models/features/molecular.py)
HammondDistanceKernel|Similarity is defined by the [Hamming distance](https://en.wikipedia.org/wiki/Hamming_distance) which considers the number of equal entries between two vectors (e.g., in One-Hot-encoding)|[Categorical](https://github.com/experimental-design/bofire/blob/main/bofire/data_models/features/categorical.py)

Note:
- Translational invariance means that the similarity between two input points is not affected by shifting both points by the same amount but only their distance determines the similarity.
- A kernel being isotropic means that the distance between two points depends only on the Euclidean distance and not on the direction in which one point lies with respect to the other.
- Rotational invariance means that the similarity measure is not affected by rotating both points by the same angle around a point filled only with zeroes. 

Note:
- SingleTaskGPSurrogate with PolynomialKernel is equivalent to PolynomialSurrogate.
- SingleTaskGPSurrogate with LinearKernel is equivalent to LinearSurrogate.
- SingleTaskGPSurrogate with TanimotoKernel is equivalent to TanimotoGP.
- One can combine two Kernels by using AdditiveKernel or MultiplicativeKernel.

### Noise model customization

For experimental data being subject to noise, one can specify the distribution of this noise. Options are:
**Noise Model**|**When to use**
:-----:|:-----:
[NormalPrior](https://github.com/experimental-design/bofire/blob/main/bofire/data_models/priors/normal.py)|Noise is Gaussian
[GammaPrior](https://github.com/experimental-design/bofire/blob/main/bofire/data_models/priors/gamma.py)|Noise has a Gamma distribution






