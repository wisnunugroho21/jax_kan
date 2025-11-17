import jax
import jax.numpy as jnp

from jax_kan.base.grid import Grid


def solve_single_lstsq(A_single: jax.Array, B_single: jax.Array) -> jax.Array:
    """
    Simulates linalg.lstsq by reformulating the problem AX = B via the normal equations: (A^T A) X = A^T B. This is used instead of linalg.lstsq because it's much faster.

    Args:
        A_single (jnp.array):
            Matrix A of AX = B, shape (M, N).
        B_single (jnp.array):
            Matrix B of AX = B, shape (M, K).

    Returns:
        single_solution (jnp.array):
            Matrix X of AX = B, shape (N, K).

    Example:
        >>> A = jnp.array([[2.0, 1.0], [1.0, 3.0]])
        >>> B = jnp.array([[1.0], [2.0]])
        >>>
        >>> solution = solve_single_lstsq(A, B)
    """

    AtA = jnp.dot(A_single.T, A_single)
    AtB = jnp.dot(A_single.T, B_single)
    single_solution = jax.scipy.linalg.solve(AtA, AtB, assume_a="pos")

    return single_solution


def solve_full_lstsq(A_full: jax.Array, B_full: jax.Array) -> jax.Array:
    """
    Parallelizes the single case, so that the problem can be solved for matrices with dimensions higher than 2. Essentially, solve_single_lstsq and solve_full_lstsq combined are a workaround, because (unlike PyTorch for example), JAX's libraries do not support lstsq for dims > 2.

    Args:
        A_full (jnp.array):
            Matrix A of AX = B, shape (batch, M, N).
        B_full (jnp.array):
            Matrix B of AX = B, shape (batch, M, K).

    Returns:
        full_solution (jnp.array):
            Matrix X of AX = B, shape (batch, N, K).

    Example:
        >>> A = jnp.array([[[2.0, 1.0], [1.0, 3.0]], [[1.0, 2.0], [2.0, 1.0]]])
        >>> B = jnp.array([[[1.0], [2.0]], [[2.0], [3.0]]])
        >>>
        >>> solution = solve_full_lstsq(A, B)
    """

    # Define the solver for (*, ., .) dimensions
    solve_full = jax.vmap(solve_single_lstsq, in_axes=(0, 0))
    # Apply it to get back the coefficients
    full_solution = solve_full(A_full, B_full)

    return full_solution


def basis_spline(
    x: jax.Array, grid: Grid, n_in: int = 2, n_out: int = 3, k: int = 3
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
    gridItem = jnp.expand_dims(grid.item, axis=2)
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
