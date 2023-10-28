import numpy as np
import matplotlib.pyplot as plt

# Constants
hbar = 1.0  # Reduced Planck's constant (set to 1 for simplicity)
m = 1.0    # Mass of the particle (set to 1 for simplicity)
L = 1.0    # Width of the box
N = 500    # Number of grid points
dx = L/N   # Grid spacing

# Potential energy inside the box (zero everywhere)
V = np.zeros(N)

# Construct the Hamiltonian matrix using finite differences
H = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i == j:
            H[i, j] = -2
        elif abs(i-j) == 1:
            H[i, j] = 1
H = -hbar**2 / (2*m*dx**2) * H + np.diag(V)

# Solve for eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(H)

# Plot the first few eigenfunctions
plt.figure(figsize=(10, 6))
for i in range(4):
    plt.plot(np.linspace(0, L, N), eigenvectors[:, i], label=f"n={i+1}, E={eigenvalues[i]:.2f}")
plt.title("Wavefunctions of a Particle in a Box")
plt.xlabel("Position (x)")
plt.ylabel("Wavefunction (Ψ(x))")
plt.legend()
plt.grid(True)
plt.show()
