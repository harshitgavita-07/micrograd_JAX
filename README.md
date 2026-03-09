# micrograd-JAX

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![JAX](https://img.shields.io/badge/JAX-ML%20framework-orange)
![Status](https://img.shields.io/badge/status-learning%20project-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Contributions](https://img.shields.io/badge/contributions-welcome-brightgreen)

A **JAX-based exploration inspired by Andrej Karpathy’s micrograd**.

The original micrograd builds a tiny automatic differentiation engine from scratch using scalar operations.  
In this version, we explore similar ideas using **JAX**, focusing on understanding:

- gradient computation
- neural network training
- optimization behaviour
- small deep learning experiments

This repository is part of my **Machine Learning learning journey**, where I study concepts and rebuild systems from first principles.

The goal is simple:

> understand neural networks deeply by rebuilding the mechanics behind them.

Instead of treating ML frameworks as black boxes, this project experiments with **minimal implementations of neural networks using JAX**.

---

## What we are building here

This repository explores:

• rebuilding neural network training loops  
• experimenting with gradients and optimization  
• understanding automatic differentiation with JAX  
• implementing simple neural network models  

It is meant to be a **learning playground for deep learning fundamentals**.

---

## Repository purpose

This project is designed to be an **open learning space**.

If you find it useful you can:

⭐ star the repo  
🍴 fork it  
💡 suggest improvements  
📚 use it to learn ML fundamentals

# micrograd

![awww](puppy.jpg)

A tiny Autograd engine (with a bite! :)). Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API. Both are tiny, with about 100 and 50 lines of code respectively. The DAG only operates over scalar values, so e.g. we chop up each neuron into all of its individual tiny adds and multiplies. However, this is enough to build up entire deep neural nets doing binary classification, as the demo notebook shows. Potentially useful for educational purposes.

### Installation

```bash
pip install micrograd
```

### Example usage

Below is a slightly contrived example showing a number of possible supported operations:

```python
from micrograd.engine import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}')
g.backward()
print(f'{a.grad:.4f}')
print(f'{b.grad:.4f}')
```

### Training a neural net

The notebook `demo.ipynb` provides a full demo of training an 2-layer neural network (MLP) binary classifier. This is achieved by initializing a neural net from `micrograd.nn` module, implementing a simple svm "max-margin" binary classification loss and using SGD for optimization. As shown in the notebook, using a 2-layer neural net with two 16-node hidden layers we achieve the following decision boundary on the moon dataset:

![2d neuron](moon_mlp.png)

### Tracing / visualization

For added convenience, the notebook `trace_graph.ipynb` produces graphviz visualizations. E.g. this one below is of a simple 2D neuron, arrived at by calling `draw_dot` on the code below, and it shows both the data (left number in each node) and the gradient (right number in each node).

```python
from micrograd import nn
n = nn.Neuron(2)
x = [Value(1.0), Value(-2.0)]
y = n(x)
dot = draw_dot(y)
```

![2d neuron](gout.svg)

### Running tests

To run the unit tests you will have to install PyTorch, which the tests use as a reference for verifying the correctness of the calculated gradients. Then simply:

```bash
python -m pytest
```

### License

MIT
