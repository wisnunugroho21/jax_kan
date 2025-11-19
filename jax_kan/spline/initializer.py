from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from jax_kan.base.initializer import SplineInitializer
from jax_kan.spline.function import basis_spline
from jax_kan.spline.grid import SplineGrid


class DefaultInitializer(SplineInitializer):
    def __init__(self, std: float = 0.1) -> None:
        self.std = std

    def initialize(
        self,
        grid: SplineGrid,
        n_in: int = 2,
        n_out: int = 5,
        k: int = 3,
        residual: Callable[[jax.Array], jax.Array] | None = nnx.silu,
        seed: int = 42,
    ) -> tuple[jax.Array | None, jax.Array]:
        rngs = nnx.Rngs(seed)

        if residual is not None:
            c_res = nnx.initializers.glorot_uniform(in_axis=-1, out_axis=-2)(
                rngs.params(), (n_out, n_in), jnp.float32
            )
        else:
            c_res = None

        c_basis = nnx.initializers.normal(stddev=self.std)(
            rngs.params(),
            (n_in * n_out, grid.G + k),
            jnp.float32,
        )

        return c_res, c_basis


class PowerInitializer(SplineInitializer):
    def __init__(
        self,
        const_b: float = 1.0,
        pow_b1: float = 0.5,
        pow_b2: float = 0.5,
        const_r: float = 1.0,
        pow_r1: float = 0.5,
        pow_r2: float = 0.5,
    ) -> None:
        self.const_b = const_b
        self.pow_b1 = pow_b1
        self.pow_b2 = pow_b2
        self.const_r = const_r
        self.pow_r1 = pow_r1
        self.pow_r2 = pow_r2

    def initialize(
        self,
        grid: SplineGrid,
        n_in: int = 2,
        n_out: int = 5,
        k: int = 3,
        residual: Callable[[jax.Array], jax.Array] | None = nnx.silu,
        seed: int = 42,
    ) -> tuple[jax.Array | None, jax.Array]:
        rngs = nnx.Rngs(seed)

        if residual is not None:
            basis_term = grid.G + k + 1

            std_res = self.const_r / ((basis_term**self.pow_r1) * (n_in**self.pow_r2))
            c_res = nnx.initializers.normal(stddev=std_res)(
                rngs.params(), (n_out, n_in), jnp.float32
            )
        else:
            basis_term = grid.G + k
            c_res = None

        std_b = self.const_b / ((basis_term**self.pow_b1) * (n_in**self.pow_b2))
        c_basis = nnx.initializers.normal(stddev=std_b)(
            rngs.params(),
            (n_in * n_out, grid.G + k),
            jnp.float32,
        )

        return c_res, c_basis


class LecunInitializer(SplineInitializer):
    def __init__(
        self,
        distribution: str = "uniform",
        sample_size: int = 10000,
        gain: float | None = None,
    ) -> None:
        self.distrib = distribution
        self.sample_size = sample_size
        self.gain = gain

    def initialize(
        self,
        grid: SplineGrid,
        n_in: int = 2,
        n_out: int = 5,
        k: int = 3,
        residual: Callable[[jax.Array], jax.Array] | None = nnx.silu,
        seed: int = 42,
    ) -> tuple[jax.Array | None, jax.Array]:
        key = jax.random.key(seed)
        rngs = nnx.Rngs(seed)

        # Generate a sample of points
        if self.distrib == "uniform":
            sample = jax.random.uniform(
                key, shape=(self.sample_size,), minval=-1.0, maxval=1.0
            )
        elif self.distrib == "normal":
            sample = jax.random.normal(key, shape=(self.sample_size,))
        else:
            sample = jnp.zeros((self.sample_size,))

        # Finally get gain
        if self.gain is None:
            gain = sample.std().item()
        else:
            gain = self.gain

        # Extend the sample to be able to pass through basis
        sample_ext = jnp.tile(sample[:, None], (1, n_in))
        # Calculate B_m^2(x)
        y_b = basis_spline(sample_ext, grid, n_in, n_out, k)
        # Calculate the average of B_m^2(x)
        y_b_sq = y_b**2
        y_b_sq_mean = y_b_sq.mean().item()

        if residual is not None:
            # Variance equipartitioned across all terms
            scale = n_in * (grid.G + k + 1)
            # Apply the residual function
            y_res = residual(sample)
            # Calculate the average of residual^2(x)
            y_res_sq = y_res**2
            y_res_sq_mean = y_res_sq.mean().item()

            std_res = gain / jnp.sqrt(scale * y_res_sq_mean)
            c_res = nnx.initializers.normal(stddev=std_res)(
                rngs.params(), (n_out, n_in), jnp.float32
            )

        else:
            # Variance equipartitioned across G+k terms
            scale = n_in * (grid.G + k)
            c_res = None

        std_b = gain / jnp.sqrt(scale * y_b_sq_mean)
        c_basis = nnx.initializers.normal(stddev=std_b)(
            rngs.params(),
            (n_out, n_in, grid.G + k),
            jnp.float32,
        )

        return c_res, c_basis


class GlorotInitializer(SplineInitializer):
    def __init__(
        self,
        distribution: str = "uniform",
        sample_size: int = 10000,
        gain: float | None = None,
    ) -> None:
        self.distrib = distribution
        self.sample_size = sample_size
        self.gain = gain

    def initialize(
        self,
        grid: SplineGrid,
        n_in: int = 2,
        n_out: int = 5,
        k: int = 3,
        residual: Callable[[jax.Array], jax.Array] | None = nnx.silu,
        seed: int = 42,
    ) -> tuple[jax.Array | None, jax.Array]:
        key = jax.random.key(seed)
        rngs = nnx.Rngs(seed)

        # Generate a sample of points
        if self.distrib == "uniform":
            sample = jax.random.uniform(
                key, shape=(self.sample_size,), minval=-1.0, maxval=1.0
            )
        elif self.distrib == "normal":
            sample = jax.random.normal(key, shape=(self.sample_size,))
        else:
            sample = jnp.zeros((self.sample_size,))

        # Finally get gain
        if self.gain is None:
            gain = sample.std().item()
        else:
            gain = self.gain

        # Extend the sample to be able to pass through basis
        sample_ext = jnp.tile(sample[:, None], (1, n_in))

        # ------------- Basis function gradient ----------------------
        # Define a scalar version of the basis function
        def basis_scalar(x):
            return basis_spline(jnp.array([[x]]), grid, n_in, n_out, k)[0, 0, :]

        # Create a Jacobian function for the scalar wrapper
        jac_basis = jax.jacobian(basis_scalar)

        num_batches = 20
        batch_size = self.sample_size // num_batches
        grad_sq_accum = 0.0

        for i in range(num_batches):
            batch = sample[i * batch_size : (i + 1) * batch_size]
            grad_batch = jax.vmap(jac_basis)(batch)
            grad_sq_accum += (grad_batch**2).sum()

        # Calculate E[B'_m^2(x)]
        grad_b_sq_mean = grad_sq_accum / (self.sample_size * (grid.G + k))
        # ------------------------------------------------------------

        # Calculate E[B_m^2(x)]
        y_b = basis_spline(sample_ext, grid)
        y_b_sq = y_b**2
        y_b_sq_mean = y_b_sq.mean().item()

        # Deal with residual if available
        if residual is not None:
            # Variance equipartitioned across all terms
            scale_in = n_in * (grid.G + k + 1)
            scale_out = n_out * (grid.G + k + 1)

            # ------------- Residual function gradient ----------------------
            # Similar idea to the basis function
            def r(x):
                return residual(x)

            jac_res = jax.jacobian(r)

            grad_res = jax.vmap(jac_res)(sample)
            # ------------------------------------------------------------

            # Calculate E[R^2(x)]
            y_res = residual(sample)
            y_res_sq = y_res**2
            y_res_sq_mean = y_res_sq.mean().item()

            # Calculate E[R'^2(x)]
            grad_res_sq = grad_res**2
            grad_res_sq_mean = grad_res_sq.mean().item()

            std_res = gain * jnp.sqrt(
                2.0 / (scale_in * y_res_sq_mean + scale_out * grad_res_sq_mean)
            )
            c_res = nnx.initializers.normal(stddev=std_res)(
                rngs.params(), (n_out, n_in), jnp.float32
            )

        else:
            # Variance equipartitioned across G+k terms
            scale_in = n_in * (grid.G + k)
            scale_out = n_out * (grid.G + k)
            c_res = None

        std_b = gain * jnp.sqrt(
            2.0 / (scale_in * y_b_sq_mean + scale_out * grad_b_sq_mean)
        )
        c_basis = nnx.initializers.normal(stddev=std_b)(
            rngs.params(),
            (n_out, n_in, grid.G + k),
            jnp.float32,
        )

        return c_res, c_basis
