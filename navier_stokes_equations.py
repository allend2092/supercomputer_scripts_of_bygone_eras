import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx, ny = 400, 100  # Grid size
rho0 = 1.0  # Density
tau = 0.6  # Relaxation time

# Directions for D2Q9 lattice
v = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1], [0, 0]])
t = np.array([1, 1, 1, 1, 1, 1, 1, 1, 4]) / 9  # weights

f1 = rho0 * (1 - 3/2 * np.square(v).sum(axis=1, keepdims=True))
f1 = np.tile(f1.reshape((9, 1, 1)), (1, nx, ny))

obstacle = np.fromfunction(lambda x, y: ((x - nx / 4) ** 2 + (y - ny / 2) ** 2) <= (ny / 8) ** 2, (nx, ny))

def equilibrium(rho, u):
    cu = np.dot(v, u.transpose(1, 0, 2))
    usqr = u[0] ** 2 + u[1] ** 2
    feq = rho * (1 + 3 * cu + 9/2 * np.square(cu) - 3/2 * usqr).reshape((v.shape[0], nx, ny))
    return t.reshape((9, 1, 1)) * feq

# Main loop
for it in range(100):
    # Drift
    for k, vi in enumerate(v):
        f1[k] = np.roll(f1[k], vi, axis=(0, 1))
    # Reflective boundaries at obstacle
    bndry_f1 = f1[:, obstacle]
    f1[5:9, obstacle] = bndry_f1[1:5]
    f1[1:5, obstacle] = bndry_f1[5:9]
    # Calculate macroscopic variables
    rho = np.sum(f1, axis=0)
    u = np.dot(v.transpose(), f1.transpose((1, 0, 2)))
    # Inlet
    u[0, 0, :] = 1
    u[1, 0, :] = 0
    rho[0, :] = 1 / (1 - u[0, 0, :]) * (np.sum(f1[range(2, 9), 0, :], axis=0) + 2 * np.sum(f1[range(1, 3), 0, :], axis=0))
    # Collision step
    feq = equilibrium(rho, u)
    f1 = (1.0 - 1.0 / tau) * f1 + 1.0 / tau * feq
    # Stability check
    u[np.abs(u) > 1e5] = 0
    rho[np.abs(rho) > 1e5] = rho0

# Visualization
plt.imshow(np.sqrt(u[0] ** 2 + u[1] ** 2).transpose(), cmap="viridis")
plt.colorbar()
plt.show()
