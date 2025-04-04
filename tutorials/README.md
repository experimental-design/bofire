# Tutorial Notebooks

The notebooks in this folder demonstrate the usage of bofire. The are organized in the following way:

### Getting Started

`getting_started.ipynb` contains the python code of the getting started section of the  [GH pages getting started](https://experimental-design.github.io/bofire/getting_started/)

### Basic Examples

Additionally, the basic functionality such as setting up the reaction domain, defining objectives and running a bayesian optimization loop is shown in a variety of notebooks by example.

### Advanced Examples
The following notebooks show more niche use cases such as the use of a Random Forest surrogate model. Advanced examples are not necessarily better strategies, they represent more complex uses of components within the library.

### Benchmarks
The benchmark tutorials exist to easily recreate results from various papers or common studies in `bofire`.

### DOE
The DOE notebooks are used to demonstrate the usage of the traditional design of experiments algorithms implemented in `bofire`, e.g. D-optimal designs.

### Serialization
All the classes in `bofire` are serializable and can be saved to json formats. The notebooks in this folder show examples of this functionality.

## Notebook testing

Notebooks should execute fast, once the `SMOKE_TEST` environment variable is present. It'll be set to true during testing a PR. Use this to check whether it is present:

```python
SMOKE_TEST = os.environ.get("SMOKE_TEST")
if SMOKE_TEST:
    # The entire Notebook should not run longer than 120 seconds. Otherwise an Error is thrown during testing
else:
    # original notebook code can run arbitrarily long
```

## Running all the Notebooks to generate outputs

By default the notebooks are run in a temporary directory when using papermill. This means that the outputs are not saved.
If you wish to update all the outputs in a systematic manner please run the following command:

```bash
python scripts/run_tutorials.py --long --in-place
```

This will run all the notebooks in the `tutorials` folder and save the outputs in the same folder. The `--long` flag is used to run the notebooks without the `SMOKE_TEST` flag. This is useful to generate the full outputs for the tutorials. The `--in-place` flag runs the notebooks in place such that the outputs are saved to the actuual notebook file rather than a copy in a temporary directory. Please clean up the outputs after running the notebooks by running the following command:

```bash
find . -name '*.ipynb' -exec nbstripout --keep-output --extra-keys "metadata.papermill.input_path metadata.papermill.output_path" {} \;
```
