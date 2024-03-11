# Surrogate models
In Bayesian Optimization, information from previous experiments is taken into account to generate proposals for future experiments. This information is leveraged by creating a surrogate model for the blck-box function that is to be optimized based on the already available data. This surrogate model serves as an approximation to the black-box function and subsequent experimental proposals are based on it. Naturally, experimental candidates for which the surrogate model makes a promosing approximation of the output values should be chosen over ones for which this is not the case. However, since the available data might cover only a small part of the input space, it is likely to only be able to make uncertain approximations to values of the black-box function far away from the data. Therefore, the surrogate model should be able to express such a measure for uncertainty next to the approximated function value.
For the decision about which experiments to propose, one can additionally specify the acquisition function. This function determines if one rather wants to focus on exploitation, i.e., quickly approaching a close local optimum of the black-box function, or on exploration, i.e., exploring multiple regions of the input space first.
Therefore, three criteria typically determine the next experimental proposals: the value of the surrogate model, its uncertainty, and the acquisition function.

BoFire offers the following classes of surrogate models.


- empirical
- fully_bayesian
- linear
- mixed_single_task_gp
- mixed_tanimoto_gp
- mlp
- polynomial
- rf
- xgb

Defaults: all von botorchStrategy: 5/2 matern kernel with automated relevance detection and normalization of the input features is used.





