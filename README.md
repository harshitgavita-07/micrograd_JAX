# micrograd-JAX

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![JAX](https://img.shields.io/badge/JAX-Google%20Research-orange)
![Project](https://img.shields.io/badge/type-learning%20project-green)
![Status](https://img.shields.io/badge/status-active-lightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Contributions](https://img.shields.io/badge/contributions-welcome-brightgreen)

A **JAX-based exploration inspired by Andrej Karpathy’s micrograd project** from the **Neural Networks: Zero to Hero** series.

The goal of this repository is to explore how **neural networks, gradients, and optimization work under the hood** while leveraging **JAX** for modern numerical computation.

Instead of treating machine learning frameworks as black boxes, this project experiments with **minimal neural network implementations and training mechanics using JAX**.

---

# Why this project exists

Modern ML frameworks like PyTorch, TensorFlow, and JAX hide a lot of complexity behind high-level APIs.

Projects like **micrograd** are powerful because they reveal the mechanics of:

- automatic differentiation
- gradient flow
- neural network training
- optimization

This repository extends that exploration by experimenting with **JAX**, a modern framework used in machine learning research.

The objective is simple:

> Learn deep learning by rebuilding and experimenting with the systems behind it.

---

# About JAX

[JAX](https://github.com/google/jax) is an open-source numerical computing and machine learning framework developed by **Google Research**.

It provides a NumPy-like API combined with powerful features for high-performance ML and scientific computing.

Key capabilities include:

• **Automatic differentiation** for computing gradients  
• **JIT compilation** for fast execution  
• **Vectorization (vmap)** for efficient batch computation  
• **Parallel execution on GPUs and TPUs**

JAX is widely used in modern ML research because it combines **clean Python code with high-performance computation**.

In this repository, JAX is used to explore neural network training, gradient computation, and optimization behaviour in a minimal and educational setting.

---

# What this repository explores

This project acts as a **learning playground for deep learning fundamentals**.

Experiments include:

• rebuilding neural network training loops  
• experimenting with gradients and optimization  
• understanding automatic differentiation with JAX  
• implementing minimal neural network models  
• studying training behaviour and learning dynamics  

The focus is **clarity and understanding**, not building a production ML framework.

---

# Repository purpose

This repository is designed as an **open learning space**.

If you find it useful you can:

⭐ Star the repository  
🍴 Fork it  
💡 Suggest improvements  
📚 Use it as a learning reference

---

# Original Project

This repository is inspired by the original **micrograd project by Andrej Karpathy**.

Original repository:

https://github.com/karpathy/micrograd

The section below contains the **original micrograd README**, preserved without modification.

---

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
