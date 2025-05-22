# Observation-on-kmeans

Code used to produce the results for the paper An Observation of Lloyd's k-Means Algorithm in High Dimensions

## Installation

The package can be installed with pip

```bash
python -m pip install .
```

## Dependencies

This package is built on JAX. In addition, we use the NMI clustering metric and the PCA decomposition implemented in Scikit-Learn.


## Reproducing results

Notebooks with the parameters used to obtain the experiments presented in the paper, as well as the functions necessary to run them. These  are available in the directory `paper_experiments/`. The notebook also provides the functions used to generate the figures presented. The results used to generate the functions are available in the directory `results/` in the form of NumPy `.npz` files.

## License

This code is Licensed under the License GPL-3.0. See the file `LICENSE` for more information.

## Acknowledgements

(suppressed for Anonymous submission)