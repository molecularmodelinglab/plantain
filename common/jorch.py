import torch
from torch.utils._pytree import tree_map
import jax
import jax.numpy as jnp
import numpy as np
from functools import reduce, wraps

def jorch_unwrap(x):
    if isinstance(x, JorchTensor):
        return x.arr
    elif isinstance(x, torch.Tensor):
        return jnp.asarray(x.detach().cpu().numpy())
    return x

def jorch_wrap(x):
    if isinstance(x, jnp.ndarray):
        return JorchTensor(x)
    elif isinstance(x, np.ndarray):
        return JorchTensor(jnp.array(x))
    return x

def parse_args(*args, **kwargs):
    args = tree_map(jorch_unwrap, args)
    kwargs = tree_map(jorch_unwrap, kwargs)
    if "dim" in kwargs:
        kwargs["axis"] = kwargs["dim"]
        del kwargs["dim"]
    if "keepdim" in kwargs:
        kwargs["keepdims"] = kwargs["keepdim"]
        del kwargs["keepdim"]
    return args, kwargs

class JorchTensor():

    def __init__(self, arr):
        self.arr = arr

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs={}):
        if func.__name__ == "unsqueeze":
            new_func = jnp.expand_dims
        elif func.__name__ == "norm":
            new_func = jax_norm
        elif func.__name__ == "clamp_min":
            new_func = jax_clamp_min
        elif func.__name__ == "cdist":
            new_func = jax_cdist
        elif func.__name__ == "cat":
            new_func = jax_cat
        elif func.__name__ == "mul":
            new_func = lambda x, y: x*y
        elif func.__name__ == "det":
            new_func = jnp.linalg.det
        else:
            if func.__name__.startswith("linalg_"):
                mod = jnp.linalg
                name = "_".join(func.__name__.split("_")[1:])
            else:
                mod = jnp
                name = func.__name__
            new_func = getattr(mod, name)
        args, kwargs = parse_args(*args, **kwargs)
        out = new_func(*args, **kwargs)
        return tree_map(jorch_wrap, out)

    def __repr__(self):
        return f"JorchTensor({self.arr})"

    def __add__(self, other):
        return JorchTensor(self.arr + jorch_unwrap(other))

    def __radd__(self, other):
        return JorchTensor(jorch_unwrap(other) + self.arr)

    def __sub__(self, other):
        return JorchTensor(self.arr - jorch_unwrap(other))

    def __rsub__(self, other):
        return JorchTensor(jorch_unwrap(other) - self.arr)

    def __mul__(self, other):
        return JorchTensor(self.arr * jorch_unwrap(other))

    def __rmul__(self, other):
        return JorchTensor(jorch_unwrap(other) * self.arr)

    def __matmul__(self, other):
        return JorchTensor(self.arr @ jorch_unwrap(other))

    def __rmatmul__(self, other):
        return JorchTensor(jorch_unwrap(other) @ self.arr)

    def __truediv__(self, other):
        return JorchTensor(self.arr / jorch_unwrap(other))

    def __rtruediv__(self, other):
        return JorchTensor(jorch_unwrap(other) / self.arr)

    def __pow__(self, other):
        return JorchTensor(self.arr ** jorch_unwrap(other))

    def __rpow__(self, other):
        return JorchTensor(jorch_unwrap(other) ** self.arr)

    def __eq__(self, other):
        return JorchTensor(self.arr.__eq__(jorch_unwrap(other)))

    def __lt__(self, other):
        return JorchTensor(self.arr.__lt__(jorch_unwrap(other)))

    def __gt__(self, other):
        return JorchTensor(self.arr.__gt__(jorch_unwrap(other)))

    def __le__(self, other):
        return JorchTensor(self.arr.__le__(jorch_unwrap(other)))

    def __ge__(self, other):
        return JorchTensor(self.arr.__ge__(jorch_unwrap(other)))

    def __neg__(self):
        return JorchTensor(-self.arr)

    def __invert__(self):
        return JorchTensor(~self.arr)

    def __len__(self):
        return len(self.arr)

    def __bool__(self):
        return self.arr.__bool__()

    def __getitem__(self, index):
        if isinstance(index, int) and index >= len(self):
            raise IndexError
        return JorchTensor(self.arr[jorch_unwrap(index)])

    def dim(self):
        return len(self.shape)

    def __getattribute__(self, name):
        if name in ("arr", "__torch_function__", "dim"): return object.__getattribute__(self, name)
        elif name == "device":
            return 'cpu'
        elif name == "is_cuda":
            return False
        elif name == "size":
            return lambda i: self.arr.shape[i]
        elif name == "flatten":
            return lambda *args, **kwargs: JorchTensor(jax_flatten(self.arr, *args, **kwargs))
        elif name == "transpose":
            return lambda dim1, dim2: JorchTensor(self.arr.swapaxes(dim1, dim2))
        elif hasattr(self.arr, name):
            attrib = getattr(self.arr, name)
            if callable(attrib):
                def ret(*args, **kwargs):
                    args, kwargs = parse_args(*args, **kwargs)
                    return jorch_wrap(attrib(*args, **kwargs))
                return ret
            else:
                return jorch_wrap(attrib)
        elif hasattr(torch, name):
            attrib = getattr(torch, name)
            return lambda *args, **kwargs: attrib(self, *args, **kwargs)
        else:
            return object.__getattribute__(self, name)

def jax_norm(arr, p, axis, keepdims, out, dtype):
    assert p in ('fro', None)
    return jnp.linalg.norm(arr, axis=axis, keepdims=keepdims)

def jax_clamp_min(arr, eps):
    return (arr - eps)*(arr > eps) + eps

def jax_flatten(arr, start_dim=0, end_dim=-1):
    assert start_dim == 0
    new_shape = (reduce(lambda x, y: x*y, arr.shape[:end_dim+1]), *arr.shape[end_dim+1:])
    return arr.flatten().reshape(new_shape)

def jax_cdist(x, y, p, compute_mode):
    assert p == 2
    return jax.vmap(lambda x1: jax.vmap(lambda y1: jnp.linalg.norm(x1 - y1))(y))(x)

def jax_cat(to_cat, axis):
    return jnp.concatenate(to_cat, axis)

def to_jax(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        args = tree_map(jorch_wrap, args)
        kwargs = tree_map(jorch_wrap, kwargs)
        out = f(*args, **kwargs)
        return tree_map(jorch_unwrap, out)
    return wrapper