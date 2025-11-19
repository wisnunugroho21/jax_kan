from typing import Callable

import jax
from flax import nnx

from jax_kan.spline.grid import SplineGrid


class SplineInitializer:
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


class ChebyshevInitializer:
    def initialize(
        self,
        n_in: int = 2,
        n_out: int = 5,
        D: int = 3,
        use_residual: bool = True,
        add_bias: bool = True,
        seed: int = 42,
    ) -> tuple[jax.Array | None, jax.Array]:
        raise NotImplementedError
