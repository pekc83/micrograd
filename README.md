# Micrograd Rust Library

This library is a Rust implementation inspired by Andrej Karpathy's video lesson on micrograd, which was originally implemented in Python. This project implements automatic differentiation and simple neural network modules from scratch in Rust. Additionally, it includes a test to compare this implementation with the Candle library.

## Features

- **Automatic Differentiation**: Supports forward and backward propagation for scalar values.
- **Basic Neural Network Components**: Includes Neuron, Layer, and MLP (multi-layer perceptron) implementations.
- **Comparison Tests**: Tests comparing the results of this implementation with the Rust Candle library to ensure correctness.
