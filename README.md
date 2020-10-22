# Variational inference for nonlinear ordinary differential equations

This repository contains code supporting the above paper.

## Dependencies
Generic scientific Python stack: `numpy`, `scipy`, `matplotlib`, `seaborn` and (optionally, for automatic Jacobians) `sympy`. Also rquires `PyTorch`.

For `Pyro` read:
https://pyro.ai/ (we recommend installing from the source)

## Usage
To run the SIR model inference, using forward sensitivity simply use the following command:
`python SIR_example.py --num_samples 1000 --warmup_steps 500  --iterations 10000 --num_qsamples 1000`. The `ProteinTransduction_example.py` file can be run with similar arguments.

### To run with adjoint sensitivity use:
`python SIR_example.py --num_samples 1000 --warmup_steps 500  --iterations 10000 --num_qsamples 1000 --adjoint True`

### To run the lotka-Volterra model with ABC-SMC run within CPP directory:
`python setup.py build_ext -i`. Then rename the generated `*.so` file to `lvssa.so` and run `LNA_abcsmc.py`. Run `LNA_variational.py` for VI with LNA.

## Using PyTorch's VJP.
All the examples in the paper were run using `SymPy` `Lamdify` function. To use PyTorch's VJP see the `SIR_torch_jacobians.py` script.
