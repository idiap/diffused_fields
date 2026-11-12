"""
Unit tests for the tangent basis used by the boundary detection.

A normal along z used to give zero tangent vectors, which made every point of a
flat patch facing the camera look like a boundary point.
"""

import numpy as np
import pytest

from diffused_fields.manifold import Pointcloud
from diffused_fields.manifold.manifold import tangent_basis


@pytest.mark.parametrize(
    "normal",
    [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 1e-9, 1.0], [0.6, 0.0, 0.8], [1.0, 2.0, 3.0]],
)
def test_tangent_basis_is_right_handed_orthonormal(normal):
    n = np.asarray([normal], dtype=float)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    u, v = tangent_basis(n)
    frame = np.stack([u[0], v[0], n[0]], axis=1)
    np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12)
    assert np.linalg.det(frame) == pytest.approx(1.0)


def test_tangent_basis_keeps_direction_of_non_degenerate_normals():
    n = np.array([[0.6, 0.0, 0.8]])
    u, _ = tangent_basis(n)
    np.testing.assert_allclose(u[0], [0.0, -1.0, 0.0])


def test_flat_patch_facing_z_has_no_interior_boundary():
    spacing = 0.002
    grid = np.arange(-10, 11) * spacing
    x, y = np.meshgrid(grid, grid)
    vertices = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, 0.5)])
    pcloud = Pointcloud(vertices=vertices, is_cluster_points=False)
    # Set rather than estimated: Open3D's normal orientation runs Qhull, which
    # rejects an exactly planar cloud.
    pcloud.normals = np.tile([0.0, 0.0, 1.0], (len(vertices), 1))
    is_boundary = pcloud.get_boundary()

    interior = np.max(np.abs(vertices[:, :2]), axis=1) < 6 * spacing
    assert not np.any(is_boundary[interior])
    assert np.any(is_boundary[~interior])
