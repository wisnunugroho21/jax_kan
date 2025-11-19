import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import nnx
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from jax_kan.chebyshev.initializer import DefaultInitializer
from jax_kan.chebyshev.layer import ChebyshevLayer


class KAN(nnx.Module):
    def __init__(self) -> None:
        self.s1 = ChebyshevLayer(2, 5, seed=42, initializer=DefaultInitializer())
        self.s2 = ChebyshevLayer(5, 1, seed=42, initializer=DefaultInitializer())

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.s1(x)
        x = self.s2(x)

        return x


def f(x: jax.Array, y: jax.Array) -> jax.Array:
    return x**2 + 2 * jnp.exp(y)


def generate_data(minval=-1, maxval=1, num_samples=1000, seed=42):
    key = jax.random.PRNGKey(seed)
    x_key, y_key = jax.random.split(key)

    x1 = jax.random.uniform(x_key, shape=(num_samples,), minval=minval, maxval=maxval)
    x2 = jax.random.uniform(y_key, shape=(num_samples,), minval=minval, maxval=maxval)

    y = f(x1, x2).reshape(-1, 1)
    X = jnp.stack([x1, x2], axis=1)

    return X, y


seed = 42

X, y = generate_data(minval=-1, maxval=1, num_samples=1000, seed=seed)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed
)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

# Initialize a KAN model
n_in = X_train.shape[1]
n_out = y_train.shape[1]
n_hidden = 6

layer_dims = [n_in, n_hidden, n_hidden, n_out]
req_params = {"D": 5, "flavor": "exact"}

model = KAN()

# print(model)

opt_type = optax.adam(learning_rate=0.001)

optimizer = nnx.Optimizer(model, optax.adam(learning_rate=0.001), wrt=nnx.Param)


# Define train loop
@nnx.jit
def train_step(model, optimizer, X_train, y_train):
    def loss_fn(model):
        residual = model(X_train) - y_train
        loss = jnp.mean((residual) ** 2)

        return loss

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)

    return loss


# Initialize train_losses
num_epochs = 2000
train_losses = jnp.zeros((num_epochs,))

for epoch in range(num_epochs):
    # Calculate the loss
    loss = train_step(model, optimizer, X_train, y_train)

    # Append the loss
    train_losses = train_losses.at[epoch].set(loss)

plt.figure(figsize=(7, 4))

plt.plot(
    np.array(train_losses),
    label="Train Loss",
    marker="o",
    color="#25599c",
    markersize=1,
)

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.yscale("log")

plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.show()

y_pred = model(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"The MSE of the fit is {mse:.5f}")

plt.figure(figsize=(7, 4))
plt.scatter(
    y_test, y_pred, alpha=0.7, color="#a3630f", marker="x", label="Predicted vs Actual"
)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="#25599c",
    linestyle="--",
    label="Perfect Fit",
)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Model Predictions vs Ground Truth")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
