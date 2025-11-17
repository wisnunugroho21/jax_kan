from typing import Callable

import jax
from flax import nnx

from jax_kan.spline.grid import SplineGrid


class Initializer:
    def initialize(
        self,
        grid: SplineGrid,
        n_in: int = 2,
        n_out: int = 5,
        k: int = 3,
        residual: Callable[[jax.Array], jax.Array] | None = nnx.silu,
        seed: int = 42,
    ) -> tuple[jax.Array | None, jax.Array]:
        raise NotImplementedError
