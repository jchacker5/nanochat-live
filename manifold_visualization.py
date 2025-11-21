import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define grid for hyperbolic + toroidal manifold simulation
x = np.linspace(-10, 10, 100)  # Periodicity (toroidal phase)
y = np.linspace(-10, 10, 100)  # Hierarchy (hyperbolic depth)
X, Y = np.meshgrid(x, y)

# Surface: Simple saddle for hyperbolic, with wave for resonance
Z = np.sin(np.sqrt(X**2 + Y**2)) / np.sqrt(X**2 + Y**2 + 1e-5) * 5 + np.cos(X / 2) * np.sin(Y / 2)

# Geodesic path: Curved trajectory from high to low energy
t = np.linspace(0, np.pi * 2, 100)
path_x = 8 * np.cos(t)
path_y = 8 * np.sin(t)
path_z = Z[np.argmin(np.abs(Y[:,0] - path_y[:,None]), axis=1), np.argmin(np.abs(X[0,:] - path_x[:,None]), axis=1)] + 0.5  # Offset for visibility

# Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the manifold surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, edgecolor='none')

# Plot the geodesic path
ax.plot(path_x, path_y, path_z, color='red', linewidth=2, label='Geodesic Path')

# Labels and title
ax.set_xlabel('X (Toroidal Phase)')
ax.set_ylabel('Y (Hyperbolic Depth)')
ax.set_zlabel('Z (Energy)')
ax.set_title('Hyperbolic + Toroidal Manifold with Geodesic Path')
ax.legend()

# Add colorbar
fig.colorbar(surf, shrink=0.5, aspect=5)

# Save the figure
plt.savefig('/home/user/nanochat-live/manifold_visualization.png', dpi=300, bbox_inches='tight')
print("Graph saved as 'manifold_visualization.png'")

# Close the plot to free memory
plt.close()
