from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=["D", "add_bias"])
def basis_chebyshev(x: jax.Array, D: int = 5, add_bias: bool = True) -> jax.Array:
    """
    Based on the degree and flavor, the values of the Chebyshev basis functions are calculated on the input.

    Args:
        x (jnp.array):
            Inputs, shape (batch, n_in).

    Returns:
        cheb (jnp.array):
            Chebyshev basis functions applied on inputs, shape (batch, n_in, D+1).

    Example:
        >>> layer = ChebyshevLayer(n_in = 2, n_out = 5, D = 5, flavor = "default",
        >>>                        residual = None, external_weights = False, init_scheme = None,
        >>>                        add_bias = True, seed = 42)
        >>>
        >>> key = jax.random.key(42)
        >>> x_batch = jax.random.uniform(key, shape=(100, 2), minval=-1.0, maxval=1.0)
        >>>
        >>> output = layer.basis(x_batch)
    """

    # Apply tanh activation
    x = jnp.tanh(x)  # (batch, n_in)

    x = jnp.expand_dims(x, axis=-1)  # (batch, n_in, 1)
    x = jnp.tile(x, (1, 1, D + 1))  # (batch, n_in, D+1)
    x = jnp.arccos(x)  # (batch, n_in, D+1)
    x *= jnp.arange(D + 1)  # (batch, n_in, D+1)
    cheb = jnp.cos(x)  # (batch, n_in, D+1)

    # Exclude the constant "1" dimension if bias is included
    if add_bias:
        return cheb[:, :, 1:]
    else:
        return cheb
