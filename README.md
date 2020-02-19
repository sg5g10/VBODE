# Black-box variational inference for nonlinear ordinary differential equations

This repository contains notebooks that demonstrate the application of variational inference for ODE models. Inference is carried using probabilistic programming platforms, `PyMC3` and `Pyro`.

To make use of probabilistic programming we have embedded the `scipy.integrate.odeint` solver in the automatic differentiation software that `PyMC3` and `Pyro` uses: i.e. `Theano` and `PyTorch`. Through these notebooks we have shown various ways towads this embedding. Thus, this technique can be easily applied to other ODE solvers such as `diffeqpy` for example. Also, domain specific (such as ecology, synthetic biology, physiology etc) modelling software can be integrated with PPLs using this technique.

## Background
Familiarity with MCMC, ODE sensititivity analysis, inverse problems and automatic differentiation (or knowledge of Backpropagation algorithm) would be useful.

## Dependencies
Generic scientific Python stack: `numpy`, `scipy`, `matplotlib`, `pandas`, `seaborn` and (optionally, for automatic Jacobians) `sympy`.

To install `PyMC3` read the following:
https://docs.pymc.io/

For `Pyro` read:
https://pyro.ai/ (we recommend installing from the source)

## Anonymity during review cycle
Note that for the double-blind review process, some outputs/messages within a code-block inside one of these notebooks have been removed. Howvere, this doesn't hamper the flow of these notebooks and all of these are fully reproducible.

## Usage
These notebooks are written in a tutorial fashion and thus we strongly recommend to follow this ordering.
Materials covered are as follows, with the next three being used in the paper:

1) Use the notebook `Lotka Volterra ForwardSens` to carry out the analysis with this model as in the paper. This shows how to write a custom `op` in `Theano` for use with `PyMC3`, using forward sensitivity analysis. NB: Use this to learn how to tackle initial values as parameters.
2) Use the notebook `Goodwin Oscillator ForwardSens`, for the Goodwin model analysis with forward sensitivity. This demonstrates embedding in `Pytorch` and subsequent use of `Pyro`.
3) `Goodwin Oscillator Adjoint` is same as above but using the adjoint sensitivity analysis.

### Automatic Jacobian
In all the above notebooks we manually define the Jacobians. However, for larger systems this can be laborious. Thus, the following notebooks show how `sympy`'s `Lambdify` function can be used to take advantage of its strong CAS. NB: ALternative to this is using automatic differentiation, but for large system (and a solution on a dense time grid) this can considerably slow down the run-time and thus we have avoided this approach.

4) `Lotka Volterra AutoJac` repeats the Lotka-Volterra problem, with forward senstivity analysis, using Lambdified Jacobians. Follow this notebook to learn this technique in the context of `Theano`, `PyMC3`.
5) `Goodwin Oscillator AutoJac` showcases this for the Goodwin model, and `Pyro` PPL.

## (Experimental) TensorFlow Probability: 
1) The `experimental` directory contains an early example of mean-field ADVI on the Fitzhugh-Nagumo model using `tfp`. However, this notebook is not polished and not been updated to use latest `TensorFlow`/`TensorFlow Probability`. However, it demonstrates the use of `py_func` to embedd `scipy`'s `odeint` in `TensorFlow`.

## TODO: 
1) Custom `C++` `op` in `PyTorch` using `boost` `odeint` or `SUNDIALS` `CVODE`.

