import jax
from flax import nnx


class KanLayer(nnx.Module):
    def basis(self, x: jax.Array) -> jax.Array:
        raise NotImplementedError

    def update_grid(self, x: jax.Array, G_new: int) -> None:
        raise NotImplementedError

    def __call__(self, x: jax.Array) -> jax.Array:
        raise NotImplementedError
