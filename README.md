# Deep function approximation

Testing various neural networks as function approximators.

## Results

The quick summary is that the MLP approximates most functions well,
including multiplication, squaring, logarithm, sin and cos.
A MLP does _not_ handle:
- reciprocal (1/x)
- quotient (x/y)
- Computing a linear gradient ((y2 - y1) / (x2 - x1))
- Linear interpolation (finding the zero of a linear function given two sample points)
- Tan

This largely seems to stem from a difficulty with singularities.

## MLFlow

Experiments are tracked and logged using MLFlow.
Start the tracking server with:
```commandline
mlflow server --host 127.0.0.1 --port 8080
```
