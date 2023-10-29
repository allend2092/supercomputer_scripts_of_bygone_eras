import torch
import pygame

# Parameters
num_particles = 1000
box_size = 500  # Adjusted for visualization
dt = 0.01
steps = 1000
damping = 0.99  # Damping factor to prevent velocities from exploding

# Initialize positions and velocities
positions = box_size * torch.rand((num_particles, 2)).cuda()
velocities = torch.randn((num_particles, 2)).cuda()

# Lennard-Jones potential
def compute_forces(positions):
    dx = positions[:, None, 0] - positions[:, 0]
    dy = positions[:, None, 1] - positions[:, 1]
    r = torch.sqrt(dx**2 + dy**2)
    r = torch.where(r < 1e-5, torch.ones_like(r).cuda() * 1e-5, r)  # Avoid division by zero
    F = 24 * ((2 * (1/r)**13) - (1/r)**7)
    Fx = F * (dx/r)
    Fy = F * (dy/r)
    return Fx.sum(dim=1), Fy.sum(dim=1)

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((box_size, box_size))
pygame.display.set_caption('Molecular Dynamics Simulation')

# MD simulation loop with pygame visualization
running = True
for _ in range(steps):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if not running:
        break

    screen.fill((255, 255, 255))  # Fill screen with white

    Fx, Fy = compute_forces(positions)
    velocities[:, 0] += Fx * dt
    velocities[:, 1] += Fy * dt
    velocities *= damping  # Apply damping to velocities
    positions += velocities * dt
    positions %= box_size  # Wrap around the box

    # Check for NaN values and reset them
    nan_indices = torch.isnan(positions).any(dim=1)
    positions[nan_indices] = box_size * torch.rand((nan_indices.sum(), 2)).cuda()

    # Draw particles
    for pos in positions:
        pygame.draw.circle(screen, (0, 0, 255), (int(pos[0].item()), int(pos[1].item())), 2)

    pygame.display.flip()
    pygame.time.wait(10)  # Delay to make visualization smoother

pygame.quit()
