from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from jax_kan.base.initializer import ChebyshevInitializer
from jax_kan.chebyshev.function import basis_chebyshev
from jax_kan.chebyshev.initializer import DefaultInitializer
from jax_kan.util.general import solve_full_lstsq


class ChebyshevLayer(nnx.Module):
    """
    ChebyshevLayer class. Corresponds to the Chebyshev version of KANs and comes in three "flavors":
        "default": the version presented in https://arxiv.org/pdf/2405.07200
        "modified": the version presented in https://www.sciencedirect.com/science/article/pii/S0045782524005462
        "exact": uses pre-defined functions for higher efficiency, but cannot scale up to arbitrary degrees

    Attributes:
        n_in (int):
            Number of layer's incoming nodes.
        n_out (int):
            Number of layer's outgoing nodes.
        D (int):
            Degree of Chebyshev polynomial (1st kind).
        flavor (Union[str, None]):
            One of "default", "modified", or "exact" - chooses basis implementation.
        residual (Union[nnx.Module, None]):
            Function that is applied on samples to calculate residual activation.
        external_weights (bool):
            Boolean that controls if the trainable weights (n_out, n_in) should be applied to the activations.
        init_scheme (Union[dict, None]):
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
        D: int = 5,
        residual: Callable[[jax.Array], jax.Array] | None = nnx.silu,
        external_weights: bool = False,
        initializer: ChebyshevInitializer = DefaultInitializer(),
        add_bias: bool = True,
        seed: int = 42,
    ):
        """
        Initializes a ChebyshevLayer instance.

        Args:
            n_in (int):
                Number of layer's incoming nodes.
            n_out (int):
                Number of layer's outgoing nodes.
            D (int):
                Degree of Chebyshev polynomial (1st kind).
            flavor (Union[str, None]):
                One of "default", "modified", or "exact" - chooses basis implementation.
            residual (Union[nnx.Module, None]):
                Function that is applied on samples to calculate residual activation.
            external_weights (bool):
                Boolean that controls if the trainable weights (n_out, n_in) should be applied to the activations.
            init_scheme (Union[dict, None]):
                Dictionary that defines how the trainable parameters of the layer are initialized.
            add_bias (bool):
                Boolean that controls wether bias terms are also included during the forward pass or not.
            seed (int):
                Random key selection for initializations wherever necessary.

        Example:
            >>> layer = ChebyshevLayer(n_in = 2, n_out = 5, D = 5, flavor = "default",
            >>>                        residual = None, external_weights = False, init_scheme = None,
            >>>                        add_bias = True, seed = 42)
        """

        # Setup basic parameters
        self.n_in = n_in
        self.n_out = n_out
        self.D = D
        self.residual = residual

        # Setup nnx rngs
        self.rngs = nnx.Rngs(seed)

        # Add bias
        if add_bias:
            self.bias = nnx.Param(jnp.zeros((n_out,)))
        else:
            self.bias = None

        # If external_weights == True, we initialize weights for the activation functions equal to unity
        if external_weights:
            self.c_ext = nnx.Param(
                nnx.initializers.ones(
                    self.rngs.params(), (self.n_out, self.n_in), jnp.float32
                )
            )
        else:
            self.c_ext = None

        # Initialize the remaining trainable parameters, based on the selected initialization scheme
        c_res, c_basis = initializer.initialize(
            n_in, n_out, D, residual is not None, add_bias, seed
        )

        self.c_basis = nnx.Param(c_basis)

        if c_res is not None:
            self.c_res = nnx.Param(c_res)
        else:
            self.c_res = None

    def update_grid(self, x, D_new):
        """
        For the case of ChebyKANs there is no concept of grid. However, a fine-graining approach can be followed by progressively increasing the degree of the polynomials.

        Args:
            x (jnp.array):
                Inputs, shape (batch, n_in).
            D_new (int):
                New Chebyshev polynomial degree.

        Example:
            >>> layer = ChebyshevLayer(n_in = 2, n_out = 5, D = 5, flavor = "default",
            >>>                        residual = None, external_weights = False, init_scheme = None,
            >>>                        add_bias = True, seed = 42)
            >>>
            >>> key = jax.random.key(42)
            >>> x_batch = jax.random.uniform(key, shape=(100, 2), minval=-1.0, maxval=1.0)
            >>>
            >>> layer.update_grid(x=x_batch, D_new=8)
        """

        # Apply the inputs to the current grid to acquire y = Sum(ciBi(x)), where ci are
        # the current coefficients and Bi(x) are the current Chebyshev basis functions
        Bi = basis_chebyshev(x, self.D, self.bias is not None).transpose(
            1, 0, 2
        )  # (n_in, batch, D+1)
        ci = self.c_basis.value.transpose(1, 2, 0)  # (n_in, D+1, n_out)
        ciBi = jnp.einsum("ijk,ikm->ijm", Bi, ci)  # (n_in, batch, n_out)

        # Update the degree order
        self.D = D_new

        # Get the Bj(x) for the degree order
        Bj = basis_chebyshev(x, self.D, self.bias is not None).transpose(
            1, 0, 2
        )  # (n_in, batch, D_new+1)

        # Solve for the new coefficients
        cj = solve_full_lstsq(Bj, ciBi)  # (n_in, D_new+1, n_out)
        # Cast into shape (n_out, n_in, D_new+1)
        cj = cj.transpose(2, 0, 1)

        self.c_basis = nnx.Param(cj)

    def __call__(self, x):
        """
        The layer's forward pass.

        Args:
            x (jnp.array):
                Inputs, shape (batch, n_in).

        Returns:
            y (jnp.array):
                Output of the forward pass, shape (batch, n_out).

        Example:
            >>> layer = ChebyshevLayer(n_in = 2, n_out = 5, D = 5, flavor = "default",
            >>>                        residual = None, external_weights = False, init_scheme = None,
            >>>                        add_bias = True, seed = 42)
            >>>
            >>> key = jax.random.key(42)
            >>> x_batch = jax.random.uniform(key, shape=(100, 2), minval=-1.0, maxval=1.0)
            >>>
            >>> output = layer(x_batch)
        """

        batch = x.shape[0]

        # Calculate basis activations
        Bi = basis_chebyshev(x, self.D, self.bias is not None)  # (batch, n_in, D+1)
        act = Bi.reshape(batch, -1)  # (batch, n_in * (D+1))

        # Check if external_weights == True
        if self.c_ext is not None:
            act_w = self.c_basis.value * self.c_ext[..., None]  # (n_out, n_in, D+1)
        else:
            act_w = self.c_basis.value

        # Calculate coefficients
        act_w = act_w.reshape(self.n_out, -1)  # (n_out, n_in * (D+1))

        y = jnp.matmul(act, act_w.T)  # (batch, n_out)

        # Check if there is a residual function
        if self.residual is not None and self.c_res is not None:
            # Calculate residual activation
            res = self.residual(x)  # (batch, n_in)
            # Multiply by trainable weights
            res_w = self.c_res.value  # (n_out, n_in)
            full_res = jnp.matmul(res, res_w.T)  # (batch, n_out)

            y += full_res  # (batch, n_out)

        if self.bias is not None:
            y += self.bias.value  # (batch, n_out)

        return y
