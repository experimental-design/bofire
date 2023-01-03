# Introduction

BoFire is a framework to define and solve black-box optimization problems. 
These problems can arise in a number of closely related fields including experimental design, multiobjective optimization and active learning.

BoFire problem specifications are json serializable for use in RESTful APIs and are to a large extent agnostic to the specific methods and frameworks in which the problems are solved.

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

## Multiobjective optimization
In the context of multiobjective optimization BoFire allows to define a vector-valued optimization problem

$$
\min_{x \in \mathbb{X}} s(y(x))
$$

where

* $x \in \mathbb{X}$ is again the experimental design space
* $y = \{y_1, \ldots y_M\}$ are known functions describing your experimental outputs and
* $s = \{s_1, \ldots s_M\}$ are the objectives to be minimized, e.g. $s_1$ is the identity function if $y_1$ is to be minimized.

Since the objectives are in general conflicting, there is no point $x$ that simulataneously optimizes all objectives.
Instead the goal is to find the Pareto front of all optimal compromises.
A decision maker can then explore these compromises to get a deep understanding of the problem and make the best informed decision.

## Bayesian optimization
In the context of Bayesian optimization we want to simultaneously learn the unknown function $y(x)$ (exploration), while focusing the experimental effort on promising regions (exploitation).
This is done by using the experimental data to fit a probabilistic model $p(y|x, {data})$ that estimates the distribution of posible outcomes for $y$.
An acquisition function $a$ then formulates the desired trade-off between exploration and exploitation

$$
\min_{x \in \mathbb{X}} a(s(p_y(x)))
$$

and the minimizer $x_\mathrm{opt}$ of this acquisition function. determines the next experiment $y(x)$ to run.
When are multiple competing objectives, the task is again to find a suitable approximation of the Pareto front.