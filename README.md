<div align="center">

# ⚡ micrograd-JAX

### Andrej Karpathy's micrograd — Rebuilt with JAX

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![JAX](https://img.shields.io/badge/JAX-Google%20Research-FF6F00?style=for-the-badge&logo=google&logoColor=white)](https://github.com/google/jax)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Stars](https://img.shields.io/github/stars/harshitgavita-07/micrograd_JAX?style=for-the-badge&color=yellow)](https://github.com/harshitgavita-07/micrograd_JAX/stargazers)

*If you understand autodiff deeply enough to rebuild it — you understand deep learning.*

</div>

---

## 🧠 What Is This?

Karpathy's [micrograd](https://github.com/karpathy/micrograd) is one of the most celebrated ML teaching tools ever written — a tiny scalar autograd engine that reveals exactly how backpropagation works.

**This repo rebuilds micrograd using JAX** — replacing the scalar computation graph with JAX's functional transforms: `grad`, `jit`, and `vmap`. The result is a minimal autograd engine that's both pedagogically clear and production-ready with GPU/TPU acceleration.

> **Why JAX?** Because modern ML research runs on JAX (DeepMind, Google Brain, many others). Understanding how JAX's autodiff works at this level is a superpower.

---

## ⚔️ micrograd vs micrograd-JAX

| Feature | Original micrograd | micrograd-JAX (this repo) |
|---|---|---|
| Engine | Custom scalar Value graph | JAX functional transforms |
| Differentiation | Manual backprop | `jax.grad` |
| Acceleration | CPU only | CPU / GPU / TPU via `jit` |
| Vectorization | None | `vmap` for batch ops |
| JIT compilation | None | `@jax.jit` |
| Research-ready | ❌ | ✅ |

---

## 🔬 Core Concepts Demonstrated

- **Automatic Differentiation** — how `grad` computes exact gradients via forward/reverse mode AD
- **JIT Compilation** — how `jit` traces and compiles a Python function to XLA
- **Vectorization** — how `vmap` eliminates explicit for-loops over batches
- **Functional Purity** — why JAX requires pure functions and how to work with it
- **Neural Net Training** — MLP trained on the moons dataset, all from scratch

---

## 🚀 Quick Start

```bash
git clone https://github.com/harshitgavita-07/micrograd_JAX.git
cd micrograd_JAX
pip install jax jaxlib numpy matplotlib
jupyter notebook demo.ipynb
```

---

## 📁 Structure

```
micrograd_JAX/
├── Mine version(JAX)_micrograd/   # JAX reimplementation
│   ├── demo.ipynb                 # Full walkthrough notebook
│   └── trace_graph.ipynb          # Computation graph visualization
├── micrograd/                     # Original Karpathy implementation (reference)
├── test/                          # Test suite
└── setup.py
```

---

## 📊 Training Result

The MLP trained on the `make_moons` dataset achieves clean decision boundary separation:

![Training Result](moon_mlp.png)

---

## 💡 Key Insight

The biggest lesson building this: **JAX doesn't have a computation graph you can inspect like micrograd's `Value` class.** Instead, JAX traces Python functions at the *type* level and generates XLA computations. This forced a deeper understanding of what autodiff actually is — not a graph, but a program transformation.

---

## 🔗 Related

- [Karpathy's micrograd](https://github.com/karpathy/micrograd) — the original
- [Neural Networks: Zero to Hero](https://youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) — the course this is based on
- [JAX docs](https://jax.readthedocs.io) — JAX transforms reference

---

<div align="center">

**If this helped you understand JAX or autodiff, drop a ⭐ — it helps others find it.**

*Built by [Harshit Gavita](https://github.com/harshitgavita-07)*

</div>
