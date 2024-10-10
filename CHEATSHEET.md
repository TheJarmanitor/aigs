t Sheet

## Overview

- **JAX**: An advanced version of NumPy that supports vectorization, GPU acceleration, XLA (Accelerated Linear Algebra), and automatic differentiation.

## Automatic Differentiation

- Use `grad` to compute the derivative of a Python function.

```python
from jax import grad

def f(x):
    return x**2

df = grad(f)  # df(x) returns 2 * x
```

## Arrays

- In JAX, everything is an array (multi-dimensional matrix).
- Example: For a 10x10x10x10 matrix `M`, element-wise multiplication is simply `M * M`.

## Vectorization with `vmap`

- `vmap` applies a function across all elements of an axis.

```python
import jax.numpy as jnp
from jax import vmap

f = lambda x: jnp.sum(x)
vf = vmap(f)  # Applies f to each row of a matrix
```

## Just-In-Time Compilation with `jit`

- `jit` compiles functions to make them faster.

```python
from jax import jit

fjv = jit(vf)
```

## PyTrees and `tree`

- PyTrees: Any Python data structure consisting of lists, tuples, dicts, etc.
- `tree` applies functions to PyTrees.

```python
from jax import tree_util as tree

params = {"conv_layers": [{"kernel": jnp.array()}], ...}
# Apply a function to every leaf of the PyTree
updated_params = tree.tree_map(lambda x: x + 1, params)
```

- Example: Subtract gradients from parameters.

```python
params = tree.tree_map(lambda p, g: p - g, params, grads)
```

## Functional Programming Principles

- **No Side Effects**: Functions should be replaceable by the value they return.

```python
# Incorrect
def f(x):
    global y
    y += 1
    return x + y

# Correct
def f(x):
    return g(x)

def g(x):
    return x + 1
```

## Printing in JAX

- Printing inside a `vmap`-ed function prints one sample.
- Printing inside a `jit`-ed function returns a `tracer`, which is related to compilation and understanding data types and shapes, but not values.

## Advanced: `lax`

- `lax` allows for extremely clean code written at a high level of abstraction.
- Considered advanced but very powerful.

```python
import jax.lax as lax

# Example usage of lax (specific examples depend on the use case)
```

## Summary

- JAX is a powerful tool for numerical computing with advanced features like automatic differentiation, vectorization, and just-in-time compilation.
- Embrace functional programming principles to write clean and efficient code.
- Explore `lax` for high-level abstractions and advanced functionality.

# General important features:

```
jnp.reshape(x, newshape)  # reshapes array
jnp.reshape(x, (-1, 1))  # reshapes array to column vector
jnp.concatenate([x, y], axis=0)  # concatenates arrays
jnp.stack([x, y], axis=0)  # stacks arrays
jnp.split(x, indices_or_sections, axis=0)  # splits array
jnp.dot(x, y)  # dot product
jnp.matmul(x, y)  # matrix multiplication
jnp.transpose(x)  # transposes array
jnp.linalg.norm(x)  # computes norm
jnp.linalg.inv(x)  # computes inverse
x.repeat(repeats, axis=None)  # repeats array
```
