# Tutorial Notebooks

The notebooks in this folder demonstrate the usage of bofire.  Below you find a list of all notebooks including some tags of what kind of problems are discussed in it. 

| notebook 	| Kind 	| Tags 	| Description 	|
|---	|---	|---	|---	|
| GettingStarted 	| Feature Introduction 	| Cotninuous Inputs<br>Discrete Inputs<br>Categorical Inputs<br>Constraints<br>Strategies 	| The introductory GH Pages code example. 	|
| basic_examples/Reaction_Optimization_Example.ipynb 	| Optimization 	| Single Objective<br>Continuous Inputs<br>Categorical Inputs<br>Bayesian Optimization 	| An example of how to optmize reaction conditions.  	|
| basic_examples/Model_Fitting_and_analysis.ipynb 	| Regression Model<br> + Validation 	| SingleTask Gaussian Process<br>Model Cross-Validation<br>Feature Importance<br>Bayesian Optimization 	| A model fitting example. 	|
| benchmarks/001-Himmelbau.ipynb 	| Singleobjective Optimization<br><br>Benchmark 	| Continuous Inputs<br>SOBO 	| An example of how to use the built-in benchmark functionality	|
| benchmarks/002-DTLZ2.ipynb 	| Multiobjective Optimization<br><br>Benchmark 	| Continuous Inputs<br>QEHVI<br>Custom Model Setup 	| An example of how to use the built-in benchmark functionality 	|
| benchmarks/003-CrossCoupling.ipynb 	| Multiobjective Optimization<br><br>Benchmark 	| Continuous Inputs<br>Categorical Inputs<br>Descriptors For Categorical Inputs<br>QPAREGO 	| An example of how to use the built-in benchmark functionality 	|
| benchmarks/004-Aspen_benchmark.ipynb 	| Multiobjective Optimization<br><br>Benchmark 	| Continuous Inputs<br>Categorical Inputs<br>Aspen<br>QNEHVI 	| An example of how to use the built-in Aspen runner to optimize digital twins	|
| benchmarks/005-Hartmann_with_nchoosek.ipynb 	| Singleobjective Optimization<br><br>Benchmark 	| Continuous Inputs<br>NChooseK constraints	| An example of how to optimize problems including NChooseK constraints	|
| benchmarks/006-30dimBranin.ipynb 	| Singleobjective Optimization<br><br>Benchmark 	| Continuous Inputs<br>Fully Bayesian	| An example of how to optimize high-dim problems with fully bayesian GPs.	|


### Getting Started

`getting_started.ipynb` contains the python code of the getting started section of the  [GH pages getting started](https://experimental-design.github.io/bofire/start)

### Basic Examples

Additionally, the basic functionality such as setting up the reaction domain, defining objectives and running a bayesian optimization loop is shown in a variety of noteboooks by example. 

## Notebook testing

Notebooks should execute fast, once the `SMOKE_TEST` environment variable is present. It'll be set to true during testing a PR. Use this to check wheter it is present:

```python
SMOKE_TEST = os.environ.get("SMOKE_TEST")
if SMOKE_TEST:
    # The entire Notebook should not run longer than 120 seconds. Otherwise an Error is thrown during testing 
else:
    # original notebook code can run arbitrarily long
```