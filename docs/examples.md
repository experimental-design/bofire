# Examples

This is a collection of code examples to allow for an easy exploration of the functionalities that BoFire offers.

## DoE

- [creating designs for constrained design spaces](basic_examples.ipynb)
- [optimizing designs with respect to various optimality criteria](optimality_criteria.ipynb)
- [creating designs for a custom model](design_with_explicit_formula.ipynb)
- [creating designs with NChooseK constraints](nchoosek_constraint.ipynb)
- [creating full and fractional factorial designs](fractional_factorial.ipynb)

## API with BoFire

You can find an examples of how BoFire can be used in APIs in separate repositories.
- The [Candidates API](https://github.com/experimental-design/bofire-candidates-api) demonstrates an API that provides get new experimental candidates based on DoE or Bayesian optimization.
- The [Types API](https://github.com/experimental-design/bofire-types-api) is an API to check serialized data models. For instance, a JavaScript frontend that allows the user to define an optimization domain can check its validity explicitly.
