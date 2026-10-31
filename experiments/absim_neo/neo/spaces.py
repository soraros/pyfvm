# SPDX-License-Identifier: GPL-3.0-or-later
"""Two native scalar spaces on a square, with explicit local incidence.

The vertex space keeps every triangle's edge contribution separately.  Thus
duplicate sparse entries remain visible to the assembly arithmetic policy.
"""

from dataclasses import dataclass
import math
import operator

import numpy as np


@dataclass(frozen=True)
class Space:
  name: str
  points: np.ndarray
  weights: np.ndarray
  edges: np.ndarray
  boundary_edges: list
  fixed_nodes: np.ndarray
  triangles: np.ndarray
  quadrature: list

  def interpolate(self, state):
    """Evaluate the native interpolant at each volume quadrature point."""
    state = np.asarray(state)
    if state.shape != (len(self.points),) or not np.isrealobj(state):
      raise ValueError("state must contain one real scalar per native unknown")
    if not np.all(np.isfinite(state)):
      raise ValueError("state must be finite")
    return np.asarray([
      math.fsum(float(state[i]) * float(phi) for i, phi in zip(ids, shape))
      for ids, shape, _, _ in self.quadrature
    ])

  def integrate(self, values):
    """Integrate already evaluated quadrature values, with one final sum."""
    values = np.asarray(values)
    if values.shape != (len(self.quadrature),) or not np.isrealobj(values):
      raise ValueError("values must contain one real scalar per quadrature point")
    if not np.all(np.isfinite(values)):
      raise ValueError("quadrature values must be finite")
    return math.fsum(
      float(value) * weight
      for value, (_, _, _, weight) in zip(values, self.quadrature)
    )


def _grid_parameters(n, length):
  if isinstance(n, (bool, np.bool_)):
    raise ValueError("n must be an integer at least one")
  try:
    n = operator.index(n)
  except TypeError as error:
    raise ValueError("n must be an integer at least one") from error
  length = float(length)
  if n < 1 or not math.isfinite(length) or length <= 0:
    raise ValueError("require n >= 1 and finite length > 0")
  area = length * length
  spacing = length / n
  if not math.isfinite(area) or area == 0 or spacing * spacing == 0:
    raise ValueError("square and cell areas must be representable in float64")
  return n, length, spacing


def cell_fv(n, length):
  """Orthogonal cell FV: midpoint fluxes and half-cell boundary distances."""
  n, length, spacing = _grid_parameters(n, length)
  axis = (np.arange(n) + 0.5) * spacing
  xx, yy = np.meshgrid(axis, axis, indexing="xy")
  points = np.column_stack((xx.ravel(), yy.ravel()))
  weights = np.full(n * n, spacing * spacing)
  edges = []
  boundary_edges = []
  for row in range(n):
    for col in range(n):
      i = row * n + col
      if col + 1 < n:
        edges.append((i, i + 1, 1.0))
      if row + 1 < n:
        edges.append((i, i + n, 1.0))
      if col == 0:
        boundary_edges.append((i, np.array([0.0, axis[row]]), 2.0))
      if col == n - 1:
        boundary_edges.append((i, np.array([length, axis[row]]), 2.0))
      if row == 0:
        boundary_edges.append((i, np.array([axis[col], 0.0]), 2.0))
      if row == n - 1:
        boundary_edges.append((i, np.array([axis[col], length]), 2.0))
  quadrature = [
    (np.array([i], dtype=int), np.array([1.0]), xy.copy(), float(weights[i]))
    for i, xy in enumerate(points)
  ]
  return Space(
    "CellFV", points, weights, np.asarray(edges, dtype=float).reshape(-1, 3),
    boundary_edges, np.empty(0, dtype=int), np.empty((0, 3), dtype=int),
    quadrature,
  )


def vertex_p1(n, length):
  """Conforming P1 triangles with degree-two volume quadrature.

  Each square uses its south-west to north-east diagonal.  Gradients are
  calculated from the actual triangle coordinates, not a graph approximation.
  """
  n, length, _ = _grid_parameters(n, length)
  axis = np.linspace(0.0, length, n + 1)
  xx, yy = np.meshgrid(axis, axis, indexing="xy")
  points = np.column_stack((xx.ravel(), yy.ravel()))
  triangles = []
  for row in range(n):
    for col in range(n):
      sw = row * (n + 1) + col
      se, nw = sw + 1, sw + n + 1
      ne = nw + 1
      triangles.extend(((sw, se, ne), (sw, ne, nw)))
  triangles = np.asarray(triangles, dtype=int)
  barycentric = np.array([
    [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
    [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
    [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
  ])
  edges, quadrature = [], []
  weight_parts = [[] for _ in points]
  for ids in triangles:
    xy = points[ids]
    a, b = xy[1] - xy[0], xy[2] - xy[0]
    twice_area = a[0] * b[1] - a[1] * b[0]
    if not math.isfinite(twice_area) or twice_area <= 0:
      raise ValueError("triangle area must be finite and positive")
    area = float(twice_area / 2.0)
    gradients = np.array([
      [xy[1, 1] - xy[2, 1], xy[2, 0] - xy[1, 0]],
      [xy[2, 1] - xy[0, 1], xy[0, 0] - xy[2, 0]],
      [xy[0, 1] - xy[1, 1], xy[1, 0] - xy[0, 0]],
    ]) / twice_area
    for i, j in ((0, 1), (1, 2), (2, 0)):
      metric = -area * float(np.dot(gradients[i], gradients[j]))
      edges.append((ids[i], ids[j], metric))
    for i in ids:
      weight_parts[i].append(area / 3.0)
    for shape in barycentric:
      quadrature.append((ids.copy(), shape.copy(), shape @ xy, area / 3.0))
  weights = np.asarray([math.fsum(parts) for parts in weight_parts])
  index = np.arange(len(points)).reshape(n + 1, n + 1)
  fixed_nodes = np.unique(np.concatenate((
    index[0], index[-1], index[:, 0], index[:, -1],
  )))
  return Space(
    "VertexP1", points, weights, np.asarray(edges, dtype=float).reshape(-1, 3),
    [], fixed_nodes, triangles, quadrature,
  )
