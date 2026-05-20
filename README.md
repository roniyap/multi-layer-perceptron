# Multi-Layer Perceptron from Scratch
**Language:** Python (no neural-network / ML libraries used)

## Overview

This project is a from-scratch implementation of a **Multi-Layer Perceptron (MLP)**
with one hidden layer, trained via the **backpropagation** algorithm. The entire
network — forward pass, backward pass, weight update, and activation functions —
is written without the use of any machine-learning or neural-network library
(e.g. no TensorFlow, PyTorch, Keras, scikit-learn). Only standard numerical
routines (NumPy for array arithmetic and random initialisation) are used.

## Goals

The MLP supports:

- Arbitrary input dimension, hidden-layer size, and output dimension
- Choice of sigmoid or linear output units (hidden units are sigmoid/tanh)
- Random small-weight initialisation
- Forward propagation to produce predictions
- Backpropagation of error and gradient accumulation
- Configurable learning rate and batch / online update schedules

## Experiments

Three experiments are run to validate the implementation:

1. **XOR** — the classic non-linearly-separable benchmark. A small MLP
   (2 inputs, 3–4 hidden units, 1 output) is trained on all four XOR
   examples and must classify them correctly.

2. **Sinusoidal regression** — 500 four-dimensional input vectors are
   generated with components drawn uniformly from `[-1, 1]`. The target is
   `sin(x1 - x2 + x3 - x4)`. The MLP is trained on 400 examples and
   evaluated on the remaining 100, comparing train vs. test error.

3. **Letter recognition (UCI)** — the
   [UCI Letter Recognition dataset](http://archive.ics.uci.edu/ml/datasets/Letter+Recognition)
   (20,000 examples, 16 numerical attributes, 26 letter classes) is split
   roughly 4:1 into train/test. The MLP has 16 inputs, a configurable
   hidden layer (~10+ units), and 26 one-hot outputs. Trained for ≥ 1000
   epochs, then evaluated by classification accuracy on the held-out set.
