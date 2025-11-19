from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from jax_kan.base.initializer import ChebyshevInitializer


class DefaultInitializer(ChebyshevInitializer):
    def __init__(self, std: float = 0.1) -> None:
        self.std = std

    def initialize(
        self,
        n_in: int = 2,
        n_out: int = 5,
        D: int = 3,
        use_residual: bool = True,
        add_bias: bool = True,
        seed: int = 42,
    ) -> tuple[jax.Array | None, jax.Array]:
        ext_dim = D if add_bias else D + 1
        rngs = nnx.Rngs(seed)

        if use_residual:
            c_res = nnx.initializers.glorot_uniform(in_axis=-1, out_axis=-2)(
                rngs.params(), (n_out, n_in), jnp.float32
            )
        else:
            c_res = None

        std = 1.0 / jnp.sqrt(n_in * ext_dim)
        c_basis = nnx.initializers.truncated_normal(stddev=std)(
            rngs.params(), (n_out, n_in, ext_dim), jnp.float32
        )

        return c_res, c_basis
