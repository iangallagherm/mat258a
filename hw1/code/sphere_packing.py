import gurobipy as gp
from gurobipy import GRB
import numpy as np

# For plotting
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

n = 2 # Number of dimensions
m = 5 # Number of spheres
r = 1 # Radius of each sphere

# Create the model
model = gp.Model()

# Create variables
x = model.addMVar(shape=(m,n), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
R = model.addVar(lb=0, name="R")

# Create the objective function
model.setObjective(R, GRB.MINIMIZE)

# Create the constraints

# Spheres don't intersect
for i in range(m):
    for j in range(i):
        # Distance between centers must be at least 2*r
        model.addConstr(
            gp.quicksum((x[i, k] - x[j, k]) ** 2 for k in range(n)) >= (2 * r) ** 2,
            name=f"no_intersection_{i}_{j}"
        )

# Spheres are inside the bounding box
for i in range(m):
    for j in range(n):
        model.addConstr(x[i, j] - r >= -R, name=f"lower_bound_{i}_{j}")
        model.addConstr(x[i, j] + r <= R, name=f"upper_bound_{i}_{j}")

# Optimize the Model
model.optimize()

# Check if the model is feasible
if model.status == GRB.OPTIMAL:
    print("Optimal solution found.")
    print(f"Minimum radius R: {R.X}")
    for i in range(m):
        print(f"Sphere {i} center: {x[i].X}")

# Create a 2d matplotlib figure of the box with the circles inside
fig, ax = plt.subplots()

x_array = np.array([x[i].X for i in range(m)])
R_val = R.X

# Plot each circle
for i in range(m):
    center = x_array[i]
    circle = Circle(center, r, fill=False, edgecolor='blue')
    ax.add_patch(circle)
    ax.plot(*center, 'bo')  # optional: plot the center as a dot

# Draw bounding square
bounding_square = Rectangle(
    (-R_val, -R_val), 2 * R_val, 2 * R_val, fill=False, edgecolor='red', linestyle='--'
)
ax.add_patch(bounding_square)

# Set limits and aspect
ax.set_xlim(-R_val - r, R_val + r)
ax.set_ylim(-R_val - r, R_val + r)
ax.set_aspect('equal')
plt.grid(True)
plt.title(f"{m} circles in square of radius {R_val:.4f}")
plt.show()
