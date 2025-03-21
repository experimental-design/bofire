# Surrogate models
In Bayesian Optimization, information from previous experiments is taken into account to generate proposals for future experiments. This information is leveraged by creating a surrogate model for the black-box function that is to be optimized based on the available data. Naturally, experimental candidates for which the surrogate model makes a promising prediction (e.g., high predicted values of a quantity we want to maximize) should be chosen over ones for which this is not the case. However, since the available data might cover only a small part of the input space, the model is likely to only be able to make very uncertain predictions far away from the data. Therefore, the surrogate model should be able to express the degree to which the predictions are uncertain so that we can use this information - combining the prediction and the associated uncertainty - to select the settings for the next experimental iteration.

The acquisition function is the object that turns the predicted distribution (you can think of this as the prediction and the prediction uncertainty) into a single quantity representing how promising a candidate experimental point seems. This function determines if one rather wants to focus on exploitation, i.e., quickly approaching a close local optimum of the black-box function, or on exploration, i.e., exploring different regions of the input space first.

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
[TanimotoGP](https://github.com/experimental-design/bofire/blob/main/bofire/surrogates/mixed_tanimoto_gp.py)|a single objective|At least one input feature is a molecule represented as fingerprint|Gaussian process on a molecule space for which Tanimoto similarity determines the similarity between points

All of these are single-objective surrogate models. For optimization of multiple objectives at the same time, a suitable [Strategy](https://github.com/experimental-design/bofire/blob/main/bofire/strategies/strategy.py) has to be chosen. Then for each objective a different surrogate model can be specified. By default the SingleTaskGPSurrogate is used.

**Example**:

    surrogate_data_0 = SingleTaskGPSurrogate(
            inputs=domain.inputs,
            outputs=Outputs(features=[domain.outputs[0]]),
    )
    surrogate_data_1 = XGBoostSurrogate(
        inputs=domain.inputs,
        outputs=Outputs(features=[domain.outputs[1]]),
    )
    qparego_data_model = QparegoStrategy(
        domain=domain,
        surrogate_specs=BotorchSurrogates(
            surrogates=[surrogate_data_0, surrogate_data_1]
        ),
    )

**Note:**

- The standard Kernel for all Gaussian Process (GP) surrogates is a 5/2 matern kernel with automated relevance detection and normalization of the input features.
- The tree-based models (RandomForestSurrogate and XGBoostSurrogate) do not have kernels but quantify uncertainty using the standard deviation of the predictions of their individual trees.
- MLP quantifies uncertainty using the standard deviation of multiple predictions that come from different dropout rates (randomly setting neural network weights to zero).

## Customization
BoFire also offers the option to customize surrogate models. In particular, it is possible to customize the SingleTaskGPSurrogate in the following ways.

### Kernel customization
Specify the [Kernel](https://github.com/experimental-design/bofire/blob/main/bofire/data_models/kernels/api.py):

**Kernel**|**Description**|**Translation invariant**|**Input variable type**
:-----:|:-----:|:-----:|:-----:
[RBFKernel](https://en.wikipedia.org/wiki/Radial_basis_function_kernel)|Based on Gaussian distribution|Yes|[Continuous](https://github.com/experimental-design/bofire/blob/main/bofire/data_models/features/continuous.py)
[MaternKernel](https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function)|Based on Gamma function; allows setting a smoothness parameter|Yes|Continuous
[PolynomialKernel](https://scikit-learn.org/stable/modules/metrics.html)|Based on dot-product of two vectors of input points|No|Continuous
[LinearKernel](https://scikit-learn.org/stable/modules/metrics.html)|Equal to dot-product of two vectors of input points|No|Continuous
TanimotoKernel|Measures similarities between binary vectors using [Tanimoto Similarity](https://en.wikipedia.org/wiki/Jaccard_index)|Not applicable|[MolecularInput](https://github.com/experimental-design/bofire/blob/main/bofire/data_models/features/molecular.py)
HammingDistanceKernel|Similarity is defined by the [Hamming distance](https://en.wikipedia.org/wiki/Hamming_distance) which considers the number of equal entries between two vectors (e.g., in One-Hot-encoding)|Not applicable|[Categorical](https://github.com/experimental-design/bofire/blob/main/bofire/data_models/features/categorical.py)

Translational invariance means that the similarity between two input points is not affected by shifting both points by the same amount but only determined by their distance. Example: with a translationally invariant kernel, the values 10 and 20 are equally similar to each other as the values 20 and 30, while with a polynomial kernel the latter pair has potentially higher similarity. Polynomial kernels are often suitable for high-dimensional inputs while for low-dimensional inputs an RBF or Matérn kernel is recommended.

**Note:**

- SingleTaskGPSurrogate with PolynomialKernel is equivalent to PolynomialSurrogate.
- SingleTaskGPSurrogate with LinearKernel is equivalent to LinearSurrogate.
- SingleTaskGPSurrogate with TanimotoKernel is equivalent to TanimotoGP.
- One can combine two Kernels by using AdditiveKernel or MultiplicativeKernel.

**Example**:

    surrogate_data_0 = SingleTaskGPSurrogate(
            inputs=domain.inputs,
            outputs=Outputs(features=[domain.outputs[0]]),
            kernel=PolynomialKernel(power=2)
    )

### Noise model customization

For experimental data subject to noise, one can specify the distribution of this noise. The [options](https://github.com/experimental-design/bofire/blob/main/bofire/data_models/priors/api.py) are:

**Noise Model**|**When to use**
:-----:|:-----:
[NormalPrior](https://github.com/experimental-design/bofire/blob/main/bofire/data_models/priors/normal.py)|Noise is Gaussian
[GammaPrior](https://github.com/experimental-design/bofire/blob/main/bofire/data_models/priors/gamma.py)|Noise has a Gamma distribution

**Example**:

    surrogate_data_0 = SingleTaskGPSurrogate(
            inputs=domain.inputs,
            outputs=Outputs(features=[domain.outputs[0]]),
            kernel=PolynomialKernel(power=2),
            noise_prior=NormalPrior(loc=0, scale=1)
    )
