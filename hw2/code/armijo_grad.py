# Gradient Descent with a fixed step size on Rosenbrock function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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


def plot_trajectory(trajectory):
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
    plt.title(f'Gradient Descent Armijo,iters={len(trajectory) - 1})')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid()
    plt.legend()
    plt.show()

def compute_armijo_s(x, a, b):
    """Compute the Armijo step size."""
    s = 1

    f, grad, _ = rosenbrock(x)
    f_next = rosenbrock(x - b**s * grad)[0]

    while f_next - f >= - a * b**s * np.dot(grad, grad):
        s += 1
        f_next = rosenbrock(x - b**s * grad)[0]

    return s


def gradient_descent_armijo(x0, a, b, tolerance, max_iter=100000):
    """Perform gradient descent with a fixed step size."""
    x = x0
    trajectory = [x.copy()]

    iter = 0
    # Compute the initial function value and gradient
    _, grad, _ = rosenbrock(x)
    while np.linalg.norm(grad) > tolerance and iter < max_iter:
        # Check the armijo condition
        s = compute_armijo_s(x, a, b)

        # Move in the direction of the negative gradient
        x -= b**s * grad
        # Update the trajectory
        trajectory.append(x.copy())

        # Compute the next values and iterate
        _, grad, _ = rosenbrock(x)
        iter += 1
    return np.array(trajectory)


x0 = np.array([-1.2, 1.0]) # Initial point
epsilon = 1e-7             # Tolerance for gradient magnitude
a = 0.8                    # Armijo parameters
b = 0.8

trajectory = gradient_descent_armijo(x0, a, b, epsilon)
plot_trajectory(trajectory)