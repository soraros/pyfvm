# SPDX-License-Identifier: GPL-3.0-or-later
"""Physical observation and local conservation contracts at awkward scales."""

from decimal import Decimal, localcontext
import math
import unittest

import numpy as np

from neo.assembly import assemble
from neo.bearing import Bearing, Parameters, pressure_increment
from neo.spaces import cell_fv, vertex_p1


class ObservationTests(unittest.TestCase):
  def test_tiny_port_increment_survives_until_explicit_publication(self):
    increment = 2.0**-40
    for constructor in (cell_fv, vertex_p1):
      with self.subTest(space=constructor.__name__):
        model = Bearing(constructor(2, 0.02))
        state = np.zeros(model.nstate)
        state[-1] = increment
        observed = model.observe(state)
        published = model.publish(state)
        p = model.parameters
        with localcontext() as context:
          context.prec = 70
          ambient = Decimal.from_float(p.ambient)
          s = Decimal.from_float(increment)
          expected = float((ambient * ambient + 2 * ambient * s).sqrt() - ambient)
        self.assertEqual(observed["port_increment_Pa"], expected)
        self.assertEqual(published["port_pressure_Pa"] - p.ambient, 0.0)
        expected_feed = 2 * p.ambient * p.porous * 0.02**2 * increment
        self.assertAlmostEqual(observed["film_feed_mass_kg_s"] / expected_feed,
                                1.0, delta=6e-16)
        restored_feed = p.porous * 0.02**2 * (
          published["port_pressure_Pa"]**2 - p.ambient**2
        )
        self.assertEqual(restored_feed, 0.0)
        self.assertGreater(observed["film_feed_mass_kg_s"], 0.0)

  def test_uniform_field_and_port_have_exactly_zero_internal_exchange(self):
    for constructor in (cell_fv, vertex_p1):
      for value in (0.0, 2.0**-40, 1e5, 2.0**40):
        with self.subTest(space=constructor.__name__, value=value):
          model = Bearing(constructor(2, 0.02))
          x = np.r_[np.full(model.nstate, value), model.parameters.gap]
          for name in ("diffusion", "feed"):
            result = assemble([model.blocks[name]], x, model.nstate)
            self.assertTrue(np.array_equal(result.residual, np.zeros(model.nstate)))
            # Differentiate the common mode, including both film and port.
            for row in result.jacobian[:, :model.nstate].toarray():
              tolerance = 4 * np.finfo(float).eps * math.fsum(abs(v) for v in row)
              self.assertLessEqual(abs(math.fsum(row)), tolerance)

  def test_unchanged_native_state_has_exactly_zero_be_storage(self):
    for constructor in (cell_fv, vertex_p1):
      with self.subTest(space=constructor.__name__):
        space = constructor(3, 0.02)
        count = len(space.points) + 1
        old = np.linspace(1e-10, 1e5, count)
        model = Bearing(space, dt=0.01, previous=old)
        x = np.r_[old, model.parameters.gap]
        for name in ("storage", "port_storage"):
          storage = assemble([model.blocks[name]], x, model.nstate).residual
          self.assertTrue(np.array_equal(storage, np.zeros(model.nstate)))

  def test_local_incidence_conservation_and_global_mass_ledger(self):
    for constructor in (cell_fv, vertex_p1):
      with self.subTest(space=constructor.__name__):
        space = constructor(4, 0.02)
        feed = lambda xy: xy[0] < 0.01 and xy[1] > 0.01
        steady = Bearing(space, feed_mask=feed)
        previous = steady.solve()
        model = Bearing(space, dt=0.02, previous=previous, feed_mask=feed)
        state = model.solve(h=9e-6)
        x = np.r_[state, 9e-6]
        for name in ("diffusion", "feed"):
          block = model.blocks[name]
          values, derivatives = block.kernel.evaluate(np.concatenate(
            [x[block.variable_ids], block.parameters], axis=1,
          ))
          self.assertTrue(np.array_equal(block.incidence[:, 0],
                                          -block.incidence[:, 1]))
          # This is a per-local-component balance, before any global rounding.
          self.assertEqual(math.fsum(np.r_[values.ravel(), -values.ravel()]), 0.0)
          self.assertEqual(math.fsum(np.r_[derivatives.ravel(), -derivatives.ravel()]), 0.0)
        observation = model.observe(state, h=9e-6)
        scale = max(abs(observation[key]) for key in (
          "outlet_mass_kg_s", "upstream_mass_kg_s", "film_storage_kg_s",
          "port_storage_kg_s",
        ))
        self.assertLess(abs(observation["balance_kg_s"]) / scale, 2e-11)
        self.assertLess(np.max(np.abs(model.evaluate(state, h=9e-6).residual)), 2e-17)

  def test_physical_domain_is_checked(self):
    with self.assertRaises(ValueError):
      pressure_increment(-101325 / 2, 101325)
    with self.assertRaises(ValueError):
      Parameters(viscosity=0)
    with self.assertRaises(ValueError):
      Bearing(cell_fv(1, 0.02), dt=0)
    model = Bearing(vertex_p1(2, 0.02), feed_mask=lambda xy: xy[0] < 0.01)
    with self.assertRaises(ValueError):
      model.exact_solution()


if __name__ == "__main__":
  unittest.main()
