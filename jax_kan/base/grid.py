import jax
import jax.numpy as jnp


class Grid:
    """
    Abstract class for the grid
    """

    def __init__(self) -> None:
        self.item = jnp.zeros(1)

    def update(self, x: jax.Array, G_new: int) -> None:
        """
        Create and initialize the grid. Can also be used to reset a grid to the default value.

        Returns:
            grid (jnp.array):
                Grid for the BaseLayer, shape (n_in*n_out, G + 2k + 1).

        Example:
            >>> grid = SplineGrid(n_in = 2, n_out = 5, k = 3, G = 3, grid_range = (-1,1), grid_e = 0.05)
            >>> grid.item = grid._initialize()
        """

        raise NotImplementedError
