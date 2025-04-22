# Gradient Descent with a fixed step size on Rosenbrock function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_trajectory(trajectory, h_k):
    """Plot the trajectory of the optimization process."""
    plt.figure(figsize=(10, 8))
    x1 = np.linspace(-4, 4, 800)
    x2 = np.linspace(-4, 4, 800)
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
    plt.title(f'Newton step_size={h_k},iters={len(trajectory) - 1})')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid()
    plt.legend()
    plt.show()


def rosenbrock(x):
    x1, x2 = x[0], x[1]
    """Compute the Rosenbrock function, its gradient, and hessian"""
    f = 100 * (x2 - x1**2)**2 + (1 - x1)**2
    grad = np.array([
        - 400 * x1 * (x2 - x1**2) - 2 * (1 - x1),
        200 * (x2 - x1**2)
                   ])
    hessian = np.array([
        [- 400 * (x2 - 3*x1**2) + 2, -400 * x1],
        [- 400 * x1, 200]
                       ])
    return f, grad, hessian


def newton_fixed_step(x0, lambda_k, tolerance, max_iter=1e6):
    """Newton's method with a fixed step size."""
    x = x0
    trajectory = [x.copy()]

    iter = 0
    # Compute the initial function value and gradient
    _, grad, hessian = rosenbrock(x)
    while np.linalg.norm(grad) > tolerance and iter < max_iter:
        # Move in the computed Newton direction
        x -= lambda_k * np.linalg.solve(hessian, grad)
        # Update the trajectory
        trajectory.append(x.copy())

        # Compute the next values and iterate
        _, grad, hessian = rosenbrock(x)
        iter += 1
    return np.array(trajectory)


x0 = np.array([-1.2, 1.0]) # Initial point
epsilon = 1e-7             # Tolerance for gradient magnitude
lambda_k = 1.0             # Step size

trajectory = newton_fixed_step(x0, lambda_k, epsilon)
plot_trajectory(trajectory, lambda_k)