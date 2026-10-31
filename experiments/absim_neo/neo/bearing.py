# SPDX-License-Identifier: GPL-3.0-or-later
"""An isothermal gas film, distributed feed and finite pressure port.

The state is s=(p**2-p0**2)/(2*p0), in Pa.  All equations and observations
retain this centred coordinate; adding p0 is an explicit publication step.
"""

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from scipy.sparse.linalg import spsolve

from .assembly import Block, assemble
from .forms import components, sqrt, symbols
from .kernels import Kernel


@dataclass(frozen=True)
class Parameters:
  ambient: float = 101325.0
  supply: float = 200000.0
  gap: float = 8e-6
  viscosity: float = 1.81e-5
  gas_rt: float = 287.05 * 293.15
  porous: float = 1e-13
  upstream: float = 4e-17
  port_volume: float = 1e-8

  def __post_init__(self):
    for name, value in vars(self).items():
      if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")

  @property
  def supply_coordinate(self):
    return (self.supply - self.ambient) * (self.supply + self.ambient) / (
      2 * self.ambient
    )


@dataclass
class Evaluation:
  residual: np.ndarray
  jacobian: object
  gap_derivative: np.ndarray
  raw_residual: np.ndarray


def pressure_increment(s, ambient):
  """Rationalised p-p0, without forming an absolute pressure."""
  s = np.asarray(s, dtype=float)
  if not np.all(np.isfinite(s)) or np.any(s <= -ambient / 2):
    raise ValueError("s must describe finite, strictly positive pressure")
  return 2 * s / (np.sqrt(1 + 2 * s / ambient) + 1)


@lru_cache(maxsize=2)
def _kernels(width):
  left, right, h, metric, p0, mu, rt = symbols("left right h metric p0 mu rt")
  conductance = h**3 * p0 / (12 * mu * rt)
  edge = Kernel(components(conductance * metric * (left - right)),
    variables=(left, right, h), parameters=(metric, p0, mu, rt), name="film_edge")
  boundary = Kernel(components(conductance * metric * left),
    variables=(left, h), parameters=(metric, p0, mu, rt), name="film_boundary")

  field = symbols(" ".join(f"s{i}" for i in range(width)))
  port, porous, weight, test, n1, n2 = symbols("port porous weight test n1 n2")
  shape = (n1, n2)
  coordinate = field[0]
  contrasts = [field[0] - port]
  for index in range(1, width):
    difference = shape[index - 1] * (field[index] - field[0])
    coordinate = coordinate + difference
    contrasts.append(difference)
  factor = 2 * p0 * porous * weight * test
  feed = Kernel(components(*(factor * item for item in contrasts)),
    variables=field + (port,), parameters=(p0, porous, weight, test, n1, n2),
    name=f"feed_{width}")

  previous, previous_h, dt = symbols("previous previous_h dt")
  old_field = symbols(" ".join(f"old{i}" for i in range(width)))
  old_coordinate = old_field[0]
  # Subtract at native DOFs before interpolation; retain each temporal part.
  # A tiny nodal change can disappear in either rounded quadrature value.
  delta0 = field[0] - old_field[0]
  temporal_parts = [delta0]
  for index in range(1, width):
    old_coordinate = old_coordinate + shape[index - 1] * (
      old_field[index] - old_field[0]
    )
    temporal_parts.append(shape[index - 1] * (
      (field[index] - old_field[index]) - delta0
    ))
  root = sqrt(1 + 2 * coordinate / p0)
  old_root = sqrt(1 + 2 * old_coordinate / p0)
  old_increment = 2 * old_coordinate / (old_root + 1)
  scale = weight * test / (rt * dt)
  temporal_scale = scale * h * 2 / (root + old_root)
  storage = Kernel(components(
    scale * p0 * (h - previous_h),
    scale * (h - previous_h) * old_increment,
    *(temporal_scale * part for part in temporal_parts),
  ), variables=field + (h,),
    parameters=(weight, test, p0, rt, dt, previous_h, n1, n2) + old_field,
    name=f"storage_{width}")

  upstream, supply, volume = symbols("upstream supply volume")
  supply_kernel = Kernel(components(2 * p0 * upstream * (port - supply)),
    variables=(port,), parameters=(p0, upstream, supply), name="upstream")
  port_storage = Kernel(components(
    volume / (rt * dt) * 2 * (port - previous) / (
      sqrt(1 + 2 * port / p0) + sqrt(1 + 2 * previous / p0)
    )
  ), variables=(port,), parameters=(volume, rt, dt, p0, previous),
    name="port_storage")
  load = Kernel(components(weight * 2 * coordinate / (root + 1)),
    variables=field, parameters=(weight, p0, n1, n2), name=f"load_{width}")
  port_pressure = Kernel(components(2 * port / (sqrt(1 + 2 * port / p0) + 1)),
    variables=(port,), parameters=(p0,), name="port_pressure")
  return dict(edge=edge, boundary=boundary, feed=feed, storage=storage,
    supply=supply_kernel, port_storage=port_storage, load=load,
    port_pressure=port_pressure)


def _block(kernel, ids, parameters, rows, incidence):
  return Block(kernel, np.asarray(ids, dtype=int),
    np.asarray(parameters, dtype=float), np.asarray(rows, dtype=int),
    np.asarray(incidence, dtype=float))


class Bearing:
  """Steady or one backward-Euler step on a native cell or vertex space.

  Only x=L is grounded; the other three sides are sealed.  The film and
  upstream restrictions are deliberately simple squared-pressure conductances.
  No choking, porous depth, moving mesh or transient history manager is implied.
  """

  def __init__(self, space, parameters=None, dt=None, previous=None,
    previous_gap=None, feed_mask=None):
    self.space = space
    self.parameters = parameters or Parameters()
    self.nfield = len(space.points)
    self.nstate = self.nfield + 1
    self.port = self.nfield
    self.gap_column = self.nstate
    self.dt = dt
    self.feed_mask = feed_mask
    self.previous = np.zeros(self.nstate) if previous is None else np.array(
      previous, dtype=float, copy=True
    )
    self.previous_gap = self.parameters.gap if previous_gap is None else previous_gap
    if self.previous.shape != (self.nstate,):
      raise ValueError("previous must have one value per field DOF plus the port")
    pressure_increment(self.previous, self.parameters.ambient)
    if not math.isfinite(self.previous_gap) or self.previous_gap <= 0:
      raise ValueError("previous_gap must be finite and positive")
    if dt is not None and (not math.isfinite(dt) or dt <= 0):
      raise ValueError("dt must be finite and positive")
    if space.boundary_edges:
      self.length = max(float(xy[0]) for _, xy, _ in space.boundary_edges)
      self.fixed = np.empty(0, dtype=int)
    else:
      self.length = float(np.max(space.points[:, 0]))
      self.fixed = np.flatnonzero(space.points[:, 0] == self.length)
    width = len(space.quadrature[0][0])
    if width not in (1, 3):
      raise ValueError("this demonstrator supports cell and triangular P1 spaces")
    self.kernels = _kernels(width)
    self.blocks = self._build_blocks()
    self.iterations = 0

  def _build_blocks(self):
    p, k = self.parameters, self.kernels
    blocks = {}
    if len(self.space.edges):
      edges = self.space.edges
      ij = edges[:, :2].astype(int)
      blocks["diffusion"] = _block(k["edge"],
        np.column_stack((ij, np.full(len(ij), self.gap_column))),
        [[e[2], p.ambient, p.viscosity, p.gas_rt] for e in edges], ij,
        np.tile([1.0, -1.0], (len(ij), 1)))
    right = [(i, metric) for i, xy, metric in self.space.boundary_edges
      if xy[0] == self.length]
    if right:
      blocks["boundary"] = _block(k["boundary"],
        [[i, self.gap_column] for i, _ in right],
        [[metric, p.ambient, p.viscosity, p.gas_rt] for _, metric in right],
        [[i] for i, _ in right], [[1.0]] * len(right))

    feed_ids, feed_params, feed_rows, storage_ids, storage_params = [], [], [], [], []
    storage_rows, load_ids, load_params = [], [], []
    for ids, shape, xy, weight in self.space.quadrature:
      n1, n2 = (shape[1], shape[2]) if len(ids) == 3 else (0.0, 0.0)
      mask = 1.0 if self.feed_mask is None else float(self.feed_mask(xy))
      if not math.isfinite(mask) or mask < 0:
        raise ValueError("feed_mask must return a finite nonnegative scale")
      load_ids.append(ids)
      load_params.append([weight, p.ambient, n1, n2])
      for node, test in zip(ids, shape):
        feed_ids.append([*ids, self.port])
        feed_params.append([p.ambient, p.porous * mask, weight, test, n1, n2])
        feed_rows.append([node, self.port])
        if self.dt is not None:
          storage_ids.append([*ids, self.gap_column])
          storage_params.append([weight, test, p.ambient, p.gas_rt, self.dt,
            self.previous_gap, n1, n2, *self.previous[ids]])
          storage_rows.append([node])
    blocks["feed"] = _block(k["feed"], feed_ids, feed_params, feed_rows,
      [[1.0, -1.0]] * len(feed_ids))
    blocks["supply"] = _block(k["supply"], [[self.port]],
      [[p.ambient, p.upstream, p.supply_coordinate]], [[self.port]], [[1.0]])
    if self.dt is not None:
      blocks["storage"] = _block(k["storage"], storage_ids, storage_params,
        storage_rows, [[1.0]] * len(storage_ids))
      blocks["port_storage"] = _block(k["port_storage"], [[self.port]],
        [[p.port_volume, p.gas_rt, self.dt, p.ambient, self.previous[self.port]]],
        [[self.port]], [[1.0]])
    self.observation_blocks = [
      _block(k["load"], load_ids, load_params, [[0]] * len(load_ids),
        [[1.0]] * len(load_ids)),
      _block(k["port_pressure"], [[self.port]], [[p.ambient]], [[1]], [[1.0]]),
    ]
    return blocks

  def _state(self, state, h):
    state = np.asarray(state, dtype=float)
    if state.shape != (self.nstate,):
      raise ValueError("state must have one value per field DOF plus the port")
    pressure_increment(state, self.parameters.ambient)
    h = self.parameters.gap if h is None else float(h)
    if not math.isfinite(h) or h <= 0:
      raise ValueError("gap must be finite and positive")
    return np.r_[state, h]

  def evaluate(self, state, h=None):
    x = self._state(state, h)
    result = assemble(list(self.blocks.values()), x, self.nstate)
    raw = result.residual.copy()
    residual = raw.copy()
    full = result.jacobian.tolil()
    for node in self.fixed:
      residual[node] = x[node]
      full.rows[node] = [int(node)]
      full.data[node] = [1.0]
    full = full.tocsr()
    return Evaluation(residual, full[:, :self.nstate],
      full[:, self.gap_column].toarray().ravel(), raw)

  def solve(self, initial=None, h=None):
    state = self.previous.copy() if initial is None else np.array(initial, dtype=float)
    self._state(state, h)
    state[self.fixed] = 0.0
    p = self.parameters
    scale = 2 * p.ambient * p.upstream * max(abs(p.supply_coordinate), p.ambient)
    for iteration in range(30):
      evaluation = self.evaluate(state, h)
      norm = float(np.max(np.abs(evaluation.residual)))
      if norm <= 1e-12 * scale:
        self.iterations = iteration
        return state
      step = spsolve(evaluation.jacobian, -evaluation.residual)
      if not np.all(np.isfinite(step)):
        raise ArithmeticError("Newton linear solve returned non-finite values")
      accepted = False
      for backtrack in range(25):
        trial = state + (0.5**backtrack) * step
        if np.any(trial <= -0.499 * p.ambient):
          continue
        trial_norm = np.max(np.abs(self.evaluate(trial, h).residual))
        if trial_norm < norm:
          state, accepted = trial, True
          break
      if not accepted:
        raise ArithmeticError("Newton backtracking could not reduce the residual")
    raise ArithmeticError("Newton failed to satisfy its mass-residual tolerance")

  def observe(self, state, h=None):
    x = self._state(state, h)
    observations = assemble(self.observation_blocks, x, 2).residual
    def block_residual(name):
      if name not in self.blocks:
        return np.zeros(self.nstate)
      return assemble([self.blocks[name]], x, self.nstate).residual
    feed = float(block_residual("feed")[self.port])
    supply = -float(block_residual("supply")[self.port])
    film_storage = math.fsum(block_residual("storage")[:self.nfield])
    port_storage = float(block_residual("port_storage")[self.port])
    if len(self.fixed):
      outlet = -math.fsum(self.evaluate(state, h).raw_residual[self.fixed])
    else:
      outlet = math.fsum(block_residual("boundary")[:self.nfield])
    return {
      "load_N": float(observations[0]),
      "port_increment_Pa": float(observations[1]),
      "upstream_mass_kg_s": supply,
      "film_feed_mass_kg_s": feed,
      "film_storage_kg_s": film_storage,
      "port_storage_kg_s": port_storage,
      "outlet_mass_kg_s": outlet,
      "balance_kg_s": math.fsum([outlet, film_storage, port_storage, -supply]),
    }

  def gap_tangent(self, state, h=None):
    evaluation = self.evaluate(state, h)
    tangent = spsolve(evaluation.jacobian, -evaluation.gap_derivative)
    rhs_norm = np.max(np.abs(evaluation.gap_derivative))
    error = np.max(np.abs(evaluation.jacobian @ tangent + evaluation.gap_derivative))
    if error > 1e-11 * rhs_norm or not np.all(np.isfinite(tangent)):
      raise ArithmeticError("implicit gap derivative failed its backward-residual gate")
    x = self._state(state, h)
    gradient = assemble(self.observation_blocks, x, 2).jacobian.getrow(0).toarray().ravel()
    derivative = math.fsum(float(a) * float(b) for a, b in zip(gradient, np.r_[tangent, 1.0]))
    return tangent, derivative

  def publish(self, state, h=None):
    """Explicitly round centred values into absolute binary64 pressures."""
    self._state(state, h)
    pressure = self.parameters.ambient + pressure_increment(state, self.parameters.ambient)
    return {"field_pressure_Pa": pressure[:self.nfield],
      "port_pressure_Pa": float(pressure[self.port]), **self.observe(state, h)}

  def exact_solution(self, h=None):
    """Continuum steady solution for this uniform strip, sampled natively."""
    if self.feed_mask is not None:
      raise ValueError("the strip oracle requires uniform distributed feed")
    p = self.parameters
    h = p.gap if h is None else float(h)
    c = h**3 * p.ambient / (12 * p.viscosity * p.gas_rt)
    beta, gamma = 2 * p.ambient * p.porous, 2 * p.ambient * p.upstream
    k = math.sqrt(beta / c)
    admittance = c * self.length * k * math.tanh(k * self.length)
    port = gamma * p.supply_coordinate / (gamma + admittance)
    field = port * (1 - np.cosh(k * self.space.points[:, 0]) / math.cosh(k * self.length))
    return field, port, admittance * port
