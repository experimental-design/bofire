<a href=https://experimental-design.github.io/bofire/>
  <img width="350" src="https://raw.githubusercontent.com/experimental-design/bofire/main/graphics/logos/bofire-long.png" alt="BoFire Logo" />
</a>

# Introduction

BoFire is a framework to define and solve black-box optimization problems.
These problems can arise in a number of closely related fields including experimental design, multi-objective optimization and active learning.

BoFire problem specifications are json serializable for use in RESTful APIs and are to a large extent agnostic to the specific methods and frameworks in which the problems are solved.

You can find code-examples in the Getting Started section of this document, as well as full worked-out examples of code-usage in the /tutorials section of this repository!

## Experimental design

In the context of experimental design BoFire allows to define a design space

$$
\mathbb{X} = x_1 \otimes x_2 \ldots \otimes x_D
$$

where the design parameters may take values depending on their type and domain, e.g.

* continuous: $x_1 \in [0, 1]$
* discrete: $x_2 \in \{1, 2, 5, 7.5\}$
* categorical: $x_3 \in \{A, B, C\}$

and a set of equations define additional experimental constraints, e.g.

* linear equality: $\sum x_i = 1$
* linear inequality: $2 x_1 \leq x_2$
* non-linear inequality: $\sum x_i^2 \leq 1$
* n-choose-k: only $k$ out of $n$ parameters can take non-zero values.

## Multi-objective optimization

In the context of multi-objective optimization BoFire allows to define a vector-valued optimization problem

$$
\mathrm{argmax}_{x \in \mathbb{X}} s(y(x))
$$

where

* $\mathbb{X}$ is again the experimental design space
* $y = \{y_1, \ldots y_M\}$ are known functions describing your experimental outputs and
* $s = \{s_1, \ldots s_M\}$ are the objectives to be maximized. For instance, $s_1$ is the identity function if $y_1$ is to be maximized.

Since the objectives are usually conflicting, there is no point $x$ that simultaneously optimizes all objectives.
Instead the goal is to find the Pareto front of all optimal compromises.

A decision maker can then explore these compromises to get a deep understanding of the problem and make the best informed decision.

## Bayesian optimization

In the context of Bayesian optimization we want to simultaneously learn the unknown function $y(x)$ (exploration), while focusing the experimental effort on promising regions (exploitation).
This is done by using the experimental data to fit a probabilistic model $p(y|x, \mathrm{data})$ that estimates the distribution of possible outcomes for $y$.
An acquisition function $a$ then formulates the desired trade-off between exploration and exploitation

$$
\mathrm{argmax}_{x \in \mathbb{X}} a(s(p_y(x)))
$$

and the maximizer $x_\mathrm{opt}$ of this acquisition function determines the next experiment $y(x)$ to run.

When there are multiple competing objectives, the task is again to find a suitable approximation of the Pareto front.

## Design of Experiments

BoFire can be used to generate optimal experimental designs with respect to various optimality criteria like D-optimality, A-optimality or uniform space filling.

For this, the user specifies a design space and a model formula, then chooses an optimality criterion and the desired number of experiments in the design. The resulting optimization problem is then solved by [IPOPT](https://coin-or.github.io/Ipopt/).

The doe subpackage also supports a wide range of constraints on the design space including linear and nonlinear equalities and inequalities as well a (limited) use of NChooseK constraints. The user can provide fixed experiments that will be treated as part of the design but remain fixed during the optimization process. While some of the *optimization* algorithms support non-continuous design variables, the doe subpackage only supports those that are continuous.

By default IPOPT uses the freely available linear solver MUMPS. For large models choosing a different linear solver (e.g. ma57 from Coin-HSL) can vastly reduce optimization time. A free academic license for Coin-HSL can be obtained [here](https://licences.stfc.ac.uk/product/coin-hsl). Instructions on how to install additional linear solvers for IPOPT are given in the [IPOPT documentation](https://coin-or.github.io/Ipopt/INSTALL.html#DOWNLOAD_HSL). For choosing a specific (HSL) linear solver in BoFire you can just pass the name of the solver to `find_local_max_ipopt()` with the `linear_solver` option together with the library's name in the option `hsllib`, e.g.
```
find_local_max_ipopt(domain, "fully-quadratic", ipopt_options={"linear_solver":"ma57", "hsllib":"libcoinhsl.so"})
```

## Reference

We would love for you to use BoFire in your work! If you do, please cite [our paper](https://arxiv.org/abs/2408.05040):

    @misc{durholt2024bofire,
      title={BoFire: Bayesian Optimization Framework Intended for Real Experiments},
      author={Johannes P. D{\"{u}}rholt and Thomas S. Asche and Johanna Kleinekorte and Gabriel Mancino-Ball and Benjamin Schiller and Simon Sung and Julian Keupp and Aaron Osburg and Toby Boyne and Ruth Misener and Rosona Eldred and Wagner Steuer Costa and Chrysoula Kappatou and Robert M. Lee and Dominik Linzner and David Walz and Niklas Wulkow and Behrang Shafei},
      year={2024},
      eprint={2408.05040},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.05040},
    }
