# Perform gradient descent with a fixed step size on Rosenbrock function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def rosenbrock(x):
    x1, x2 = x[0], x[1]
    """Compute the Rosenbrock function, its gradient, and hessian"""
    f = 100 * (x2 - x1**2)**2 + (1 - x1)**2
    grad = np.array([
        - 4 * 100 * x1 * (x2 - x1**2) - 2 * (1 - x1),
        2 * 100 * (x2 - x1**2)
                   ])
    hessian = np.array([
        [- 4 * 100 * x2 + 8 * 100 * x1**2 + 2, -4 * 100 * x1],
        [- 4 * 100 * x1, 200]
                       ])
    return f, grad, hessian


def gradient_descent_fixed_step(x0, step_size, tolerance, max_iter=100000):
    """Perform gradient descent with a fixed step size."""
    x = x0
    trajectory = [x.copy()]

    iter = 0
    # Compute the initial function value and gradient
    _, grad, _ = rosenbrock(x)
    while np.linalg.norm(grad) > tolerance and iter < max_iter:
        # Move in the direction of the negative gradient
        x -= step_size * grad
        # Update the trajectory
        trajectory.append(x.copy())

        # Compute the next values and iterate
        _, grad, _ = rosenbrock(x)
        iter += 1
    return np.array(trajectory)


def plot_trajectory(trajectory, h_k):
    """Plot the trajectory of the optimization process."""
    plt.figure(figsize=(10, 8))
    x1 = np.linspace(-2, 2, 400)
    x2 = np.linspace(-1, 4, 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.array([[rosenbrock(np.array([x1, x2]))[0] for x1 in x1] for x2 in x2])
    plt.contourf(X1, X2, Z, levels=50, cmap='viridis', norm=LogNorm())
    plt.colorbar(label='Function value')
    plt.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        color='red',
        marker='.',
        markersize=2,
        label='Trajectory')
    plt.title(f'Gradient Descent step_size={h_k},iters={len(trajectory) - 1})')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid()
    plt.legend()
    plt.show()

x0 = np.array([-1.2, 1.0]) # Initial point
epsilon = 1e-7             # Tolerance for gradient magnitude
h_k = 1e-3                 # Step size

# Perform gradient descent
trajectory = gradient_descent_fixed_step(x0, h_k, epsilon)
# Plot the trajectory
plot_trajectory(trajectory, h_k)