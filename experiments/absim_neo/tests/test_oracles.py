# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent geometry, analytic Schur and physical-storage checks.

These references do not use generated kernels, Space.edges, Space.weights or
Space.quadrature.  The P1 reference assembles full gradient and consistent-mass
matrices directly from triangle coordinates.  Non-polynomial storage and load
use the declared degree-two cubature, independently written below.
"""

from decimal import Decimal, localcontext
import math
import unittest

import numpy as np

from neo.bearing import Bearing, Parameters
from neo.assembly import assemble
from neo.spaces import cell_fv, vertex_p1


def reference_geometry(kind, n, length, feed_mask=None):
  """Return independently assembled unit diffusion, mass and cubature."""
  dx = length / n
  side = n if kind == "cell" else n + 1
  count = side * side
  diffusion = np.zeros((count, count))
  mass = np.zeros_like(diffusion)
  quadrature = []
  if kind == "cell":
    for row in range(n):
      for col in range(n):
        i = row * n + col
        point = np.array([(col + 0.5) * dx, (row + 0.5) * dx])
        if feed_mask is None or feed_mask(point):
          mass[i, i] = dx * dx
        quadrature.append((np.array([i]), np.ones(1), dx * dx))
        for j in (i + 1 if col + 1 < n else -1,
                  i + n if row + 1 < n else -1):
          if j >= 0:
            diffusion[i, i] += 1
            diffusion[j, j] += 1
            diffusion[i, j] -= 1
            diffusion[j, i] -= 1
        if col == n - 1:
          diffusion[i, i] += 2
    return diffusion, mass, quadrature, np.empty(0, dtype=int)
  nodes = np.array([(j * dx, i * dx) for i in range(side) for j in range(side)])
  shapes = np.array([[2 / 3, 1 / 6, 1 / 6],
                      [1 / 6, 2 / 3, 1 / 6],
                      [1 / 6, 1 / 6, 2 / 3]])
  for row in range(n):
    for col in range(n):
      a = row * side + col
      for indices in ((a, a + 1, a + side + 1),
                      (a, a + side + 1, a + side)):
        ids = np.array(indices)
        vandermonde = np.column_stack((np.ones(3), nodes[ids]))
        area = abs(np.linalg.det(vandermonde)) / 2
        gradients = np.linalg.inv(vandermonde)[1:]
        diffusion[np.ix_(ids, ids)] += area * gradients.T @ gradients
        if feed_mask is None or feed_mask(nodes[ids].mean(axis=0)):
          mass[np.ix_(ids, ids)] += area / 12 * (np.ones((3, 3)) + np.eye(3))
        quadrature.extend((ids, shape, area / 3) for shape in shapes)
  fixed = np.arange(n, count, side)
  return diffusion, mass, quadrature, fixed


def steady_oracle(kind, n, length, parameters, gap=None, feed_mask=None):
  """Solve the scalar port Schur formula, independently of Newton assembly."""
  p = parameters
  h = p.gap if gap is None else gap
  diffusion, mass, quadrature, fixed = reference_geometry(kind, n, length, feed_mask)
  count = len(mass)
  coefficient = h**3 * p.ambient / (12 * p.viscosity * p.gas_rt)
  beta, gamma = 2 * p.ambient * p.porous, 2 * p.ambient * p.upstream
  a = beta * mass @ np.ones(count)
  d = float(np.sum(a))
  matrix = coefficient * diffusion + beta * mass
  free = np.setdiff1d(np.arange(count), fixed)
  reduced = matrix[np.ix_(free, free)]
  v = np.linalg.solve(reduced, a[free])
  conductance = d - a[free] @ v
  supply = (p.supply - p.ambient) * (p.supply + p.ambient) / (2 * p.ambient)
  port = gamma * supply / (gamma + conductance)
  state = np.zeros(count + 1)
  state[free], state[-1] = v * port, port
  kh = 3 * coefficient / h * diffusion[np.ix_(free, free)]
  conductance_h = v @ kh @ v
  port_h = -port * conductance_h / (gamma + conductance)
  tangent = np.zeros_like(state)
  tangent[free] = -np.linalg.solve(reduced, kh @ v) * port + v * port_h
  tangent[-1] = port_h
  jacobian = np.block([[matrix, -a[:, None]],
                        [-a[None, :], np.array([[d + gamma]])]])
  jacobian[fixed] = 0
  jacobian[fixed, fixed] = 1
  return state, tangent, jacobian, conductance * port, quadrature


def pressure_observation(state, tangent, quadrature, ambient):
  load, derivative = [], []
  for ids, shape, weight in quadrature:
    s = float(shape @ state[ids])
    physical = math.sqrt(ambient * ambient + 2 * ambient * s)
    load.append(weight * (physical - ambient))
    derivative.append(weight * ambient / physical * float(shape @ tangent[ids]))
  return math.fsum(load), math.fsum(derivative)


def transient_oracle(kind, n, length, parameters, state, previous, gap, old_gap, dt,
                      feed_mask=None):
  """Differentiate the physical p*h law by hand and evaluate it in Decimal."""
  p = parameters
  diffusion, mass, quadrature, fixed = reference_geometry(kind, n, length, feed_mask)
  count = len(mass)
  coefficient = gap**3 * p.ambient / (12 * p.viscosity * p.gas_rt)
  beta, gamma = 2 * p.ambient * p.porous, 2 * p.ambient * p.upstream
  a = beta * mass @ np.ones(count)
  matrix = coefficient * diffusion + beta * mass
  jacobian = np.block([[matrix, -a[:, None]],
                        [-a[None, :], np.array([[np.sum(a) + gamma]])]])
  gap_derivative = np.r_[3 * coefficient / gap * diffusion @ state[:-1], 0.0]
  storage = np.zeros(count + 1)
  factor = 1 / (p.gas_rt * dt)
  with localcontext() as context:
    context.prec = 70
    dec = lambda x: Decimal.from_float(float(x))
    datum = dec(p.ambient)
    for ids, shape, weight in quadrature:
      s, old_s = float(shape @ state[ids]), float(shape @ previous[ids])
      pressure = (datum * datum + 2 * datum * dec(s)).sqrt()
      old_pressure = (datum * datum + 2 * datum * dec(old_s)).sqrt()
      inventory = dec(weight) * (dec(gap) * pressure - dec(old_gap) * old_pressure)
      contribution = float(inventory / (dec(p.gas_rt) * dec(dt)))
      storage[ids] += shape * contribution
      jacobian[np.ix_(ids, ids)] += (
        factor * weight * gap * p.ambient / float(pressure) * np.outer(shape, shape)
      )
      gap_derivative[ids] += factor * weight * float(pressure) * shape
    port_p = (datum * datum + 2 * datum * dec(state[-1])).sqrt()
    old_port_p = (datum * datum + 2 * datum * dec(previous[-1])).sqrt()
    storage[-1] = float(dec(p.port_volume) * (port_p - old_port_p)
                        / (dec(p.gas_rt) * dec(dt)))
    jacobian[-1, -1] += factor * p.port_volume * p.ambient / float(port_p)
  supply = (p.supply - p.ambient) * (p.supply + p.ambient) / (2 * p.ambient)
  residual = np.r_[matrix @ state[:-1] - a * state[-1],
                    (np.sum(a) + gamma) * state[-1] - a @ state[:-1] - gamma * supply]
  residual += storage
  raw = residual.copy()
  residual[fixed] = state[fixed]
  jacobian[fixed] = 0
  jacobian[fixed, fixed] = 1
  gap_derivative[fixed] = 0
  return residual, jacobian, gap_derivative, raw


class IndependentOracles(unittest.TestCase):
  def test_steady_state_and_gap_sensitivity_match_independent_schur(self):
    for kind, constructor in (("cell", cell_fv), ("p1", vertex_p1)):
      with self.subTest(space=kind):
        n, length, p = 4, 0.01, Parameters()
        model = Bearing(constructor(n, length), p)
        state = model.solve()
        expected, tangent, jacobian, flow, quadrature = steady_oracle(kind, n, length, p)
        np.testing.assert_allclose(state, expected, rtol=2e-10, atol=2e-7)
        np.testing.assert_allclose(model.evaluate(state).jacobian.toarray(), jacobian,
                                    rtol=2e-13, atol=2e-25)
        actual_tangent, actual_load_tangent = model.gap_tangent(state)
        np.testing.assert_allclose(actual_tangent, tangent, rtol=3e-10, atol=1e-2)
        load, load_tangent = pressure_observation(expected, tangent, quadrature, p.ambient)
        observation = model.observe(state)
        self.assertAlmostEqual(observation["load_N"] / load, 1.0, delta=2e-10)
        self.assertAlmostEqual(actual_load_tangent / load_tangent, 1.0, delta=3e-10)
        self.assertAlmostEqual(observation["upstream_mass_kg_s"] / flow, 1.0, delta=3e-10)

  def test_both_native_spaces_refine_to_continuum_port_solution(self):
    p, length = Parameters(), 0.01
    coefficient = p.gap**3 * p.ambient / (12 * p.viscosity * p.gas_rt)
    k = math.sqrt(2 * p.ambient * p.porous / coefficient)
    conductance = coefficient * length * k * math.tanh(k * length)
    gamma = 2 * p.ambient * p.upstream
    supply = (p.supply - p.ambient) * (p.supply + p.ambient) / (2 * p.ambient)
    port = gamma * supply / (gamma + conductance)
    for constructor in (cell_fv, vertex_p1):
      errors = []
      for n in (4, 8):
        model = Bearing(constructor(n, length), p)
        state = model.solve()
        errors.append(abs(state[-1] / port - 1))
      with self.subTest(space=constructor.__name__):
        self.assertLess(errors[1], errors[0] * 0.4)
        self.assertLess(errors[1], 0.002)

  def test_transient_jacobian_and_storage_match_physical_law(self):
    for kind, constructor in (("cell", cell_fv), ("p1", vertex_p1)):
      with self.subTest(space=kind):
        n, length, p = 4, 0.01, Parameters()
        feed = lambda point: point[0] < length / 2 and point[1] > length / 2
        previous = steady_oracle(kind, n, length, p, feed_mask=feed)[0]
        gap, dt = 9e-6, 0.02
        model = Bearing(constructor(n, length), p, dt=dt, previous=previous,
                        previous_gap=p.gap, feed_mask=feed)
        state = model.solve(h=gap)
        residual, jacobian, direct, raw = transient_oracle(
          kind, n, length, p, state, previous, gap, p.gap, dt, feed_mask=feed,
        )
        actual = model.evaluate(state, h=gap)
        np.testing.assert_allclose(actual.residual, residual, rtol=2e-8, atol=2e-21)
        np.testing.assert_allclose(actual.raw_residual, raw, rtol=2e-9, atol=2e-21)
        np.testing.assert_allclose(actual.jacobian.toarray(), jacobian,
                                    rtol=5e-13, atol=2e-25)
        np.testing.assert_allclose(actual.gap_derivative, direct, rtol=5e-13, atol=1e-16)
        self.assertLess(np.max(np.abs(residual)), 2e-17)
        direction = np.linspace(-0.6, 0.7, len(state)) * 1e4
        step = 2e-4
        finite = (model.evaluate(state + step * direction, h=gap).residual
                  - model.evaluate(state - step * direction, h=gap).residual) / (2 * step)
        np.testing.assert_allclose(finite, jacobian @ direction, rtol=2e-7, atol=1e-17)
        gap_step = 2e-5 * gap
        finite_gap = (model.evaluate(state, h=gap + gap_step).residual
                      - model.evaluate(state, h=gap - gap_step).residual) / (2 * gap_step)
        np.testing.assert_allclose(finite_gap, direct, rtol=2e-8, atol=1e-13)
        expected_tangent = np.linalg.solve(jacobian, -direct)
        tangent, _ = model.gap_tangent(state, h=gap)
        np.testing.assert_allclose(tangent, expected_tangent, rtol=2e-10, atol=1e-2)

  def test_asymmetric_partial_feed_matches_independent_matrix(self):
    n, length, p = 4, 0.01, Parameters()
    feed = lambda point: point[0] < length / 2 and point[1] > length / 2
    for kind, constructor in (("cell", cell_fv), ("p1", vertex_p1)):
      with self.subTest(space=kind):
        model = Bearing(constructor(n, length), p, feed_mask=feed)
        state = model.solve()
        expected, tangent, _, flow, _ = steady_oracle(
          kind, n, length, p, feed_mask=feed,
        )
        np.testing.assert_allclose(state, expected, rtol=2e-10, atol=2e-7)
        actual_tangent = model.gap_tangent(state)[0]
        np.testing.assert_allclose(actual_tangent, tangent, rtol=4e-10, atol=1e-2)
        side = n if kind == "cell" else n + 1
        field = state[:-1].reshape(side, side)
        self.assertGreater(np.max(np.ptp(field, axis=0)), 100.0)
        self.assertGreater(np.max(np.ptp(field, axis=1)), 100.0)
        observation = model.observe(state)
        self.assertAlmostEqual(observation["upstream_mass_kg_s"] / flow, 1.0, delta=3e-10)
        self.assertLess(abs(observation["balance_kg_s"]), 1e-17)

  def test_gap_tangent_matches_independent_resolves(self):
    for constructor in (cell_fv, vertex_p1):
      with self.subTest(space=constructor.__name__):
        p = Parameters()
        model = Bearing(constructor(3, 0.01), p)
        state = model.solve()
        tangent, load_tangent = model.gap_tangent(state)
        epsilon = 2e-5 * p.gap
        plus, minus = model.solve(h=p.gap + epsilon), model.solve(h=p.gap - epsilon)
        finite = (plus - minus) / (2 * epsilon)
        np.testing.assert_allclose(tangent, finite, rtol=2e-7, atol=1e-2)
        finite_load = (model.observe(plus, h=p.gap + epsilon)["load_N"]
                        - model.observe(minus, h=p.gap - epsilon)["load_N"]) / (2 * epsilon)
        self.assertAlmostEqual(load_tangent / finite_load, 1.0, delta=3e-7)

  def test_p1_load_integrates_native_pressure_before_averaging(self):
    p, length = Parameters(), 0.01
    space = vertex_p1(1, length)
    model = Bearing(space, p)
    state = np.r_[2e5 * space.points[:, 0] / length, 0.0]
    quadrature = reference_geometry("p1", 1, length)[2]
    expected = pressure_observation(state, np.zeros_like(state), quadrature, p.ambient)[0]
    load = model.observe(state)["load_N"]
    collapsed = length**2 * (math.sqrt(p.ambient**2 + 2 * p.ambient * 1e5) - p.ambient)
    self.assertAlmostEqual(load / expected, 1.0, delta=3e-14)
    self.assertGreater(abs(load - collapsed), 0.01 * abs(load))

  def test_be_preserves_one_ulp_native_history_increment(self):
    """Large common coordinates must not erase a small native inventory change.

    Symmetric cubature can hide the defective local rows in an accurate total.
    Check both against Decimal, and retain the premature-interpolation control.
    This is an arithmetic stress case, not a physically qualified gas pressure.
    """
    p, length, dt = Parameters(), 0.01, 0.02
    space = vertex_p1(1, length)
    previous = np.full(len(space.points) + 1, float(2**40))
    current = previous.copy()
    current[0] += math.ulp(current[0])
    model = Bearing(space, p, dt=dt, previous=previous, previous_gap=p.gap)
    actual = assemble([model.blocks["storage"]], np.r_[current, p.gap],
                      len(current)).residual[:-1]
    quadrature = reference_geometry("p1", 1, length)[2]
    late = np.zeros(len(space.points))
    with localcontext() as context:
      context.prec = 70
      dec = lambda value: Decimal.from_float(float(value))
      datum = dec(p.ambient)
      exact = [Decimal(0) for _ in space.points]
      for ids, shape, weight in quadrature:
        basis = [1 - dec(shape[1]) - dec(shape[2]), dec(shape[1]), dec(shape[2])]
        new_coordinate = sum(coefficient * dec(current[i])
                              for i, coefficient in zip(ids, basis))
        old_coordinate = sum(coefficient * dec(previous[i])
                              for i, coefficient in zip(ids, basis))
        new_pressure = (datum * datum + 2 * datum * new_coordinate).sqrt()
        old_pressure = (datum * datum + 2 * datum * old_coordinate).sqrt()
        inventory = (dec(weight) * dec(p.gap) * (new_pressure - old_pressure)
                      / (dec(p.gas_rt) * dec(dt)))
        for i, test in zip(ids, shape):
          exact[i] += dec(test) * inventory
        new_rounded, old_rounded = float(current[ids[0]]), float(previous[ids[0]])
        for j in (1, 2):
          new_rounded += shape[j] * (current[ids[j]] - current[ids[0]])
          old_rounded += shape[j] * (previous[ids[j]] - previous[ids[0]])
        lost_difference = 2 * (new_rounded - old_rounded) / (
          math.sqrt(1 + 2 * new_rounded / p.ambient)
          + math.sqrt(1 + 2 * old_rounded / p.ambient)
        )
        late[ids] += shape * weight * p.gap / (p.gas_rt * dt) * lost_difference
      expected = np.array([float(value) for value in exact])
      total = float(sum(exact))
    self.assertGreater(total, 0)
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=0)
    observation = model.observe(current)
    self.assertAlmostEqual(observation["film_storage_kg_s"] / total, 1.0, delta=3e-14)
    self.assertGreater(np.max(abs(late - expected)) / np.max(abs(expected)), 0.2)
    self.assertAlmostEqual(math.fsum(late) / total, 1.0, delta=3e-14)


if __name__ == "__main__":
  unittest.main()
