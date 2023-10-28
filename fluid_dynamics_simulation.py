import numpy as np
import pygame

# Parameters
nx, ny = 400, 100  # Grid size
rho0 = 1.0  # Density
tau = 0.6  # Relaxation time
niters = 3  # Number of iterations

# Directions for D2Q9 lattice
v = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1], [0, 0]])
t = np.array([1, 1, 1, 1, 1, 1, 1, 1, 4]) / 9  # weights

f1 = rho0 * (1 + 3 * v[:, 0] + 9 * v[:, 0]**2 / 2 - 3 / 2 * (v[:, 0]**2 + v[:, 1]**2)).reshape(9, 1, 1)
f1 = np.tile(f1, (1, nx, ny))

obstacle = np.fromfunction(lambda x, y: ((x - nx / 4)**2 + (y - ny / 2)**2) <= (ny / 8)**2, (nx, ny))

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((nx, ny))
pygame.display.set_caption('Fluid Dynamics Simulation')

def equilibrium(rho, u):
    cu = np.dot(v, u.transpose(1, 0, 2))
    usqr = u[0]**2 + u[1]**2
    feq = rho * (1 + 3 * cu + 9 * cu**2 / 2 - 3 * usqr / 2).reshape((9, nx, ny))
    return t.reshape(9, 1, 1) * feq

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Driven lid boundary condition
    u = np.zeros((2, nx, ny))
    u[0, -1, :] = 1

    # BGK update rule
    rho = np.sum(f1, axis=0)
    f1 += (1.0 / tau) * (equilibrium(rho, u) - f1)

    # Bounce-back boundary condition for obstacle
    bndry_f1 = f1[:, obstacle]
    for k, vi in enumerate(v):
        f1[3 - k, obstacle] = bndry_f1[k]

    # Streaming step
    for k, vi in enumerate(v):
        f1[k] = np.roll(f1[k], shift=vi, axis=(0, 1))

    # Visualization
    screen.fill((255, 255, 255))
    for x in range(nx):
        for y in range(ny):
            if obstacle[x, y]:
                pygame.draw.circle(screen, (0, 0, 0), (x, y), 1)
            elif rho[x, y] > 1:
                pygame.draw.circle(screen, (0, 0, 255), (x, y), 1)
            elif rho[x, y] < 0:
                pygame.draw.circle(screen, (255, 0, 0), (x, y), 1)

    pygame.display.flip()

pygame.quit()
