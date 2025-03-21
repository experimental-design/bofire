# Examples

This is a collection of code examples to allow for an easy exploration of the functionalities that BoFire offers.
We provide [even more tutorials](https://github.com/experimental-design/bofire/tree/main/tutorials) in the repository.

## DoE

- [Creating designs for constrained design spaces](basic_examples.ipynb)
- [Optimizing designs with respect to various optimality criteria](optimality_criteria.ipynb)
- [Creating designs for a custom model](design_with_explicit_formula.ipynb)
- [Creating designs with NChooseK constraints](nchoosek_constraint.ipynb)
- [Creating full and fractional factorial designs](fractional_factorial.ipynb)

## Bayesian Optimization for Chemistry

These examples show how the tools provided by BoFire can be used for Bayesian
Optimization with some of the challenges faced in real-world experiments:

- [A toy example for optimizing a reaction](reaction_optimization.ipynb)
- [Using a Tanimoto fingerprint kernel to optimize over molecules](fingerprint_bayesopt.ipynb)
- [Using a MultiFidelity strategy with cheap, approximate experiments](multifidelity_bo.ipynb)

## API with BoFire

You can find an examples of how BoFire can be used in APIs in separate repositories:

- The [Candidates API](https://github.com/experimental-design/bofire-candidates-api) demonstrates an API that provides get new experimental candidates based on DoE or Bayesian optimization.
- The [Types API](https://github.com/experimental-design/bofire-types-api) is an API to check serialized data models. For instance, a JavaScript frontend that allows the user to define an optimization domain can check its validity explicitly.
