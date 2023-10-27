import numpy as np
import pygame

# Constants
G = 6.67430e-11  # Gravitational constant
softening = 0.5  # Softening parameter

# Initialize particles in a smaller circular pattern
n_particles = 70
theta = np.linspace(0, 2 * np.pi, n_particles)
radius = 4 # Reduced radius to place particles closer to the center
positions = np.vstack((radius * np.cos(theta), radius * np.sin(theta))).T
velocities = np.zeros((n_particles, 2))
masses = np.ones(n_particles) * 1e10  # particle mass

# Initialize pygame
pygame.init()
width, height = 1024, 768
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('N-Body Simulation')
clock = pygame.time.Clock()

def compute_forces(positions, masses):
    forces = np.zeros_like(positions)
    for i in range(n_particles):
        for j in range(n_particles):
            if i != j:
                r = positions[j] - positions[i]
                r_mag = np.sqrt(np.sum(r ** 2) + softening ** 2)
                r_hat = r / r_mag
                force_mag = G * masses[i] * masses[j] / r_mag ** 2
                forces[i] += force_mag * r_hat
    return forces

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Compute forces and update positions and velocities
    forces = compute_forces(positions, masses)
    velocities += forces / masses[:, np.newaxis] * 0.0001  # Reduce the time step for velocity update
    positions += velocities

    # Clear screen
    screen.fill((0, 0, 0))

    # Draw particles
    for pos in positions:
        x = int(pos[0] * 40 + width / 2)
        y = int(pos[1] * 40 + height / 2)
        # print(x)
        # print(y)
        if 0 <= x < width and 0 <= y < height:
            pygame.draw.circle(screen, (255, 255, 255), (x, y), 3)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
