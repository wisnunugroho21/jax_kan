from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=["n_in", "n_out", "k"])
def basis_spline(
    x: jax.Array, gridItem: jax.Array, n_in: int = 2, n_out: int = 3, k: int = 3
) -> jax.Array:
    """
    Uses k and the current grid to calculate the values of spline basis functions on the input.

    Args:
        x (jnp.array):
            Inputs, shape (batch, n_in).

    Returns:
        basis_splines (jnp.array):
            Spline basis functions applied on inputs, shape (n_in*n_out, G+k, batch).

    Example:
        >>> layer = BaseLayer(n_in = 2, n_out = 5, k = 3,
        >>>                   G = 3, grid_range = (-1,1), grid_e = 0.05, residual = nnx.silu,
        >>>                   external_weights = True, init_scheme = None, add_bias = True,
        >>>                   seed = 42)
        >>>
        >>> key = jax.random.key(42)
        >>> x_batch = jax.random.uniform(key, shape=(100, 2), minval=-1.0, maxval=1.0)
        >>>
        >>> output = layer.basis(x_batch)
    """

    batch = x.shape[0]
    # Extend to shape (batch, n_in*n_out)
    x_ext = jnp.einsum(
        "ij,k->ikj",
        x,
        jnp.ones(
            n_out,
        ),
    ).reshape((batch, n_in * n_out))
    # Transpose to shape (n_in*n_out, batch)
    x_ext = jnp.transpose(x_ext, (1, 0))

    # Broadcasting for vectorized operations
    gridItem = jnp.expand_dims(gridItem, axis=2)
    x = jnp.expand_dims(x_ext, axis=1)

    # k = 0 case
    basis_splines = ((x >= gridItem[:, :-1]) & (x < gridItem[:, 1:])).astype(float)

    # Recursion done through iteration
    for K in range(1, k + 1):
        left_term = (x - gridItem[:, : -(K + 1)]) / (
            gridItem[:, K:-1] - gridItem[:, : -(K + 1)]
        )
        right_term = (gridItem[:, K + 1 :] - x) / (
            gridItem[:, K + 1 :] - gridItem[:, 1:(-K)]
        )

        basis_splines = (
            left_term * basis_splines[:, :-1] + right_term * basis_splines[:, 1:]
        )

    return basis_splines
