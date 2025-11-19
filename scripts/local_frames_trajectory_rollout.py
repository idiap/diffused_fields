"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
Creates a trajectory by following a simple action sequence on local reference frames
"""

import numpy as np

from diffused_fields.diffusion import PointcloudScalarDiffusion, WalkOnSpheresDiffusion
from diffused_fields.manifold import Pointcloud
from diffused_fields.visualization.plotting_ps import *

# Configuration
# ==========================================
filename = "spot.ply"


pcloud = Pointcloud(filename=filename)
pcloud.get_normals()


# Load teleoperation parameters and create scalar diffusion
pcloud.initial_vertex = 350
pcloud.source_vertices = [338]
pcloud.distance_to_surface = -0.06
pcloud.trajectory_rollout_steps = 8
pcloud.step_size = 0.007

scalar_diffusion = PointcloudScalarDiffusion(pcloud, diffusion_scalar=1000)
scalar_diffusion.get_local_bases()

# Calculate initial position for trajectory
initial_position = (
    pcloud.vertices[pcloud.initial_vertex]
    + pcloud.normals[pcloud.initial_vertex] * pcloud.distance_to_surface
)

# Set up boundaries and WoS diffusion solver
boundaries = [pcloud]
wos_diffusion = WalkOnSpheresDiffusion(
    boundaries=boundaries,
    convergence_threshold=pcloud.get_mean_edge_length() * 2,
)

# Define trajectory information explicitly in the script
# ====================================================


# Define the sequence of axes to follow during trajectory rollout
trajectory_axis_sequence = ["+z", "+x", "+y"]
# Alternative examples:

# Define custom direction mappings (axis_index, sign)
# axis_index: 0=x-axis, 1=y-axis, 2=z-axis
# sign: +1=positive direction, -1=negative direction
trajectory_direction_mappings = {
    "+x": [0, 1],  # Move along positive x-axis
    "-x": [0, -1],  # Move along negative x-axis
    "+y": [1, 1],  # Move along positive y-axis
    "-y": [1, -1],  # Move along negative y-axis
    "+z": [2, 1],  # Move along positive z-axis
    "-z": [2, -1],  # Move along negative z-axis
}


print(f"Trajectory configuration:")
print(f"  - Axis sequence: {trajectory_axis_sequence}")
print(f"  - Steps per axis: {pcloud.trajectory_rollout_steps}")
print(f"  - Step size: {pcloud.step_size}")
print(f"  - Direction mappings: {trajectory_direction_mappings}")

# Execute trajectory rollout
positions, local_bases = wos_diffusion.trajectory_rollout(
    initial_position=initial_position,
    steps=pcloud.trajectory_rollout_steps,
    step_size=pcloud.step_size,
    axis_sequence=trajectory_axis_sequence,
    direction_mappings=trajectory_direction_mappings,
)

# Calculate and display trajectory statistics
velocities = np.diff(positions, axis=0)
print(f"\nTrajectory results:")
print(f"  - Total positions: {len(positions)}")
print(f"  - Velocity shape: {velocities.shape}")
print(
    f"  - Average velocity magnitude: {np.mean(np.linalg.norm(velocities, axis=1)):.6f}"
)
# Visualization
ps.init()

# Plot trajectory with orientation field
plot_orientation_field(
    positions[::4],
    local_bases[::4],
    "trajectory",
    enable_vector=True,
    enable_z=True,
    vector_length=0.05,
    vector_radius=0.01,
)

# Plot point cloud with scalar diffusion field
ps_field = plot_orientation_field(pcloud.vertices, pcloud.local_bases, "pcloud")
ps_field.add_scalar_quantity(values=scalar_diffusion.ut, name="diffusion_field")

# Plot trajectory path
ps.register_curve_network(
    "trajectory_path",
    positions,
    radius=0.005,
    edges="line",
    color=[1, 0, 0],
)

ps.show()
