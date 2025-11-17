from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from jax_kan.base.initializer import Initializer
from jax_kan.base.layer import KanLayer
from jax_kan.spline.function import basis_spline, solve_full_lstsq
from jax_kan.spline.grid import SplineGrid
from jax_kan.spline.initializer import DefaultInitializer


class SplineKanLayer(KanLayer):
    """
    SplineKanLayer class. Corresponds to the original spline-based KAN Layer introduced in the original version of KAN. Ref: https://arxiv.org/abs/2404.19756

    Attributes:
        n_in (int):
            Number of layer's incoming nodes.
        n_out (int):
            Number of layer's outgoing nodes.
        k (int):
            Order of the spline basis functions.
        G (int):
            Number of grid intervals.
        grid_range (tuple):
            An initial range for the grid's ends, although adaptivity can completely change it.
        grid_e (float):
            Parameter that defines if the grids are uniform (grid_e = 1.0) or sample-dependent (grid_e = 0.0). Intermediate values correspond to a linear mixing of the two cases.
        residual (Union[nnx.Module, None]):
            Function that is applied on samples to calculate residual activation.
        weight_coef_spline (bool):
            Boolean that controls if the trainable weights of shape (n_out, n_in) applied to the splines should be used.
        initializer (Union[dict, None]):
            Dictionary that defines how the trainable parameters of the layer are initialized.
        add_bias (bool):
            Boolean that controls wether bias terms are also included during the forward pass or not.
        seed (int):
            Random key selection for initializations wherever necessary.
    """

    def __init__(
        self,
        n_in: int = 2,
        n_out: int = 5,
        k: int = 3,
        G: int = 3,
        grid_range: tuple = (-1, 1),
        grid_e: float = 0.05,
        residual: Callable[[jax.Array], jax.Array] | None = nnx.silu,
        weight_coef_spline: bool = True,
        initializer: Initializer = DefaultInitializer(),
        add_bias: bool = True,
        seed: int = 42,
    ):
        """
        Initializes a SplineKanLayer instance.

        Args:
            n_in (int):
                Number of layer's incoming nodes.
            n_out (int):
                Number of layer's outgoing nodes.
            k (int):
                Order of the spline basis functions.
            G (int):
                Number of grid intervals.
            grid_range (tuple):
                An initial range for the grid's ends, although adaptivity can completely change it.
            grid_e (float):
                Parameter that defines if the grids are uniform (grid_e = 1.0) or sample-dependent (grid_e = 0.0). Intermediate values correspond to a linear mixing of the two cases.
            residual (Union[nnx.Module, None]):
                Function that is applied on samples to calculate residual activation.
            weight_coef_spline (bool):
                Boolean that controls if the trainable weights of shape (n_out, n_in) applied to the splines should be used.
            initializer (Union[dict, None]):
                Dictionary that defines how the trainable parameters of the layer are initialized.
            add_bias (bool):
                Boolean that controls wether bias terms are also included during the forward pass or not.
            seed (int):
                Random key selection for initializations wherever necessary.

        Example:
            >>> layer = SplineKanLayer(n_in = 2, n_out = 5, k = 3,
            >>>                   G = 3, grid_range = (-1,1), grid_e = 0.05, residual = nnx.silu,
            >>>                   weight_coef_spline = True, initializer = DefaultInitializer(), add_bias = True,
            >>>                   seed = 42)
        """

        # Setup basic parameters
        self.n_in = n_in
        self.n_out = n_out
        self.k = k
        self.grid_range = grid_range
        self.residual = residual

        # Setup nnx rngs
        self.rngs = nnx.Rngs(seed)

        # Initialize the grid
        self.grid = SplineGrid(
            n_in=n_in, n_out=n_out, k=k, G=G, grid_range=grid_range, grid_e=grid_e
        )

        # If weight_coef_spline == True, we initialize weights for the spline functions equal to unity
        if weight_coef_spline:
            self.c_spl = nnx.Param(
                nnx.initializers.ones(
                    self.rngs.params(), (self.n_out, self.n_in), jnp.float32
                )
            )
        else:
            self.c_spl = None

        # Initialize the remaining trainable parameters, based on the selected initialization scheme
        c_res, c_basis = initializer.initialize(
            self.grid,
            n_in,
            n_out,
            k,
            residual,
            seed,
        )

        self.c_basis = nnx.Param(c_basis)

        if c_res is not None:
            self.c_res = nnx.Param(c_res)
        else:
            self.c_res = None

        # Add bias
        if add_bias:
            self.bias = nnx.Param(jnp.zeros((n_out,)))
        else:
            self.bias = None

    def update_grid(self, x: jax.Array, G_new: int) -> None:
        """
        Performs a grid update given a new value for G (i.e., G_new) and adapts it to the given data, x. Additionally, re-initializes the c_basis parameters to a better estimate, based on the new grid.

        Args:
            x (jnp.array):
                Inputs, shape (batch, n_in).
            G_new (int):
                Size of the new grid (in terms of intervals).

        Example:
            >>> layer = SplineKanLayer(n_in = 2, n_out = 5, k = 3,
            >>>                   G = 3, grid_range = (-1,1), grid_e = 0.05, residual = nnx.silu,
            >>>                   weight_coef_spline = True, initializer = DefaultInitializer(), add_bias = True,
            >>>                   seed = 42)
            >>>
            >>> key = jax.random.key(42)
            >>> x_batch = jax.random.uniform(key, shape=(100, 2), minval=-1.0, maxval=1.0)
            >>>
            >>> layer.update_grid(x=x_batch, G_new=5)
        """

        # Apply the inputs to the current grid to acquire y = Sum(ciBi(x)), where ci are
        # the current coefficients and Bi(x) are the current spline basis functions
        Bi = basis_spline(x, self.grid)  # (n_in*n_out, G+k, batch)
        ci = self.c_basis.value  # (n_in*n_out, G+k)
        ciBi = jnp.einsum("ij,ijk->ik", ci, Bi)  # (n_in*n_out, batch)

        # Update the grid
        self.grid.update(x, G_new)

        # Get the Bj(x) for the new grid
        A = basis_spline(x, self.grid)  # shape (n_in*n_out, G_new+k, batch)
        Bj = jnp.transpose(A, (0, 2, 1))  # shape (n_in*n_out, batch, G_new+k)

        # Expand ciBi from (n_in*n_out, batch) to (n_in*n_out, batch, 1)
        ciBi = jnp.expand_dims(ciBi, axis=-1)

        # Solve for the new coefficients
        cj = solve_full_lstsq(Bj, ciBi)
        # Cast into shape (n_in*n_out, G_new+k)
        cj = jnp.squeeze(cj, axis=-1)

        self.c_basis = nnx.Param(cj)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        The layer's forward pass.

        Args:
            x (jnp.array):
                Inputs, shape (batch, n_in).

        Returns:
            y (jnp.array):
                Output of the forward pass, corresponding to the weighted sum of the B-spline activation and the residual activation, shape (batch, n_out).

        Example:
            >>> layer = SplineKanLayer(n_in = 2, n_out = 5, k = 3,
            >>>                   G = 3, grid_range = (-1,1), grid_e = 0.05, residual = nnx.silu,
            >>>                   weight_coef_spline = True, initializer = DefaultInitializer(), add_bias = True,
            >>>                   seed = 42)
            >>>
            >>> key = jax.random.key(42)
            >>> x_batch = jax.random.uniform(key, shape=(100, 2), minval=-1.0, maxval=1.0)
            >>>
            >>> output = layer(x_batch)
        """

        batch = x.shape[0]

        # Calculate spline basis activations
        Bi = basis_spline(
            x, self.grid, self.n_in, self.n_out, self.k
        )  # (n_in*n_out, G+k, batch)
        ci = self.c_basis.value  # (n_in*n_out, G+k)
        # Calculate spline activation
        spl = jnp.einsum("ij,ijk->ik", ci, Bi)  # (n_in*n_out, batch)
        # Transpose to shape (batch, n_in*n_out)
        spl = jnp.transpose(spl, (1, 0))

        # Check if weight_coef_spline == True
        if self.c_spl is not None:
            # Reshape constants to (1, n_in*n_out)
            cnst_spl = jnp.expand_dims(self.c_spl.value, axis=0).reshape(
                (1, self.n_in * self.n_out)
            )
            # Calculate spline term
            y = cnst_spl * spl  # (batch, n_in*n_out)
        else:
            y = spl  # (batch, n_in*n_out)

        # Check if there is a residual function
        if self.residual is not None and self.c_res is not None:
            # Extend x to shape (batch, n_in*n_out)
            x_ext = jnp.einsum(
                "ij,k->ikj",
                x,
                jnp.ones(
                    self.n_out,
                ),
            ).reshape((batch, self.n_in * self.n_out))
            # Transpose to shape (n_in*n_out, batch)
            x_ext = jnp.transpose(x_ext, (1, 0))
            # Calculate residual activation - shape (batch, n_in*n_out)
            res = jnp.transpose(self.residual(x_ext), (1, 0))
            # Reshape constant to (1, n_in*n_out)
            cnst_res = jnp.expand_dims(self.c_res.value, axis=0).reshape(
                (1, self.n_in * self.n_out)
            )
            # Calculate the entire activation
            y += cnst_res * res  # (batch, n_in*n_out)

        # Reshape and sum
        y_reshaped = jnp.reshape(y, (batch, self.n_out, self.n_in))
        y = jnp.sum(y_reshaped, axis=2)  # (batch, n_out)

        if self.bias is not None:
            y += self.bias.value  # (batch, n_out)

        return y
