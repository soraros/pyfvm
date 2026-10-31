# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent controls for ordered evaluation and component transport.

Fraction oracles below operate on the exact rational values of the binary64
inputs.  They do not invoke the generated derivative or assembly algorithms.
"""

from dataclasses import FrozenInstanceError
from fractions import Fraction
from itertools import permutations
import math
import unittest

import numpy as np
from scipy.sparse import coo_matrix

from neo.assembly import Block, assemble
from neo.forms import components, fma, sqrt, symbols
from neo.kernels import Kernel


BIG = float(2**53)


def exact_weighted_sum(values, weights):
  return float(sum((
    Fraction.from_float(float(value)) * Fraction.from_float(float(weight))
    for value, weight in zip(values, weights)
  ), Fraction()))


def one_entity(kernel, ids, rows=(0,), incidence=(1.0,), parameters=()):
  return Block(
    kernel, np.asarray([ids], dtype=int), np.asarray([parameters], dtype=float),
    np.asarray([rows], dtype=int), np.asarray([incidence], dtype=float),
  )


class ComponentTests(unittest.TestCase):
  def test_authoring_is_immutable_and_keeps_operand_order(self):
    x, = symbols("x")
    original = components(x, 2 * x)
    snapshot = original.parts
    added = original + components(3 * x)
    subtracted = original - components(x * x, 3 * x)
    negated = -original
    scaled = 2 * original
    self.assertEqual(original.parts, snapshot)
    self.assertIsNot(original, added)
    with self.assertRaises(FrozenInstanceError):
      x.value = "changed"
    with self.assertRaises(FrozenInstanceError):
      original.parts = ()
    for form, expected in (
      (original, [2.0, 4.0]),
      (added, [2.0, 4.0, 6.0]),
      (subtracted, [2.0, 4.0, -4.0, -6.0]),
      (negated, [-2.0, -4.0]),
      (scaled, [4.0, 8.0]),
    ):
      with self.subTest(expected=expected):
        values, _ = Kernel(form, (x,)).evaluate([[2.0]])
        np.testing.assert_array_equal(values[0], expected)

  def test_ordered_values_and_mathematical_derivatives_are_distinct_contracts(self):
    x, a = symbols("x a")
    kernel = Kernel(components((a + x) - a, a + (x - a)), (x,), (a,))
    values, derivatives = kernel.evaluate([[1.0, BIG]])
    np.testing.assert_array_equal(values[0], [0.0, 1.0])
    np.testing.assert_array_equal(derivatives[0], [1.0, 1.0])
    self.assertEqual(kernel.derivative_variables, (0, 0))
    left = Kernel((a + x) - a, (x,), (a,))
    right = Kernel(a + (x - a), (x,), (a,))
    self.assertNotEqual(left.source, right.source)

  def test_derivative_paths_survive_an_ordered_value_cancellation(self):
    x, a = symbols("x a")
    kernel = Kernel((a * x + x) - a * x, (x,), (a,))
    block = one_entity(kernel, (0,), parameters=(BIG,))
    result = assemble([block], np.array([1.0]), 1)
    self.assertEqual(result.residual[0], 0.0)
    self.assertEqual(result.jacobian[0, 0], 1.0)
    values, derivatives = kernel.evaluate([[1.0, BIG]])
    self.assertEqual(values[0, 0], 0.0)
    self.assertEqual(math.fsum(derivatives[0]), 1.0)
    ordered_total = 0.0
    for part in derivatives[0]:
      ordered_total += float(part)
    self.assertEqual(ordered_total, 0.0)

  def test_declared_components_preserve_values_and_jacobians(self):
    x, = symbols("x")
    kernel = Kernel(components(BIG * x, x, -BIG * x), (x,))
    result = assemble([one_entity(kernel, (0,))], np.array([1.0]), 1)
    self.assertEqual(result.residual[0], 1.0)
    self.assertEqual(result.jacobian[0, 0], 1.0)
    self.assertTrue(result.jacobian.has_canonical_format)
    self.assertEqual(result.jacobian.nnz, 1)

  def test_components_cross_entity_boundaries_without_a_local_subtotal(self):
    x, a, b = symbols("x a b")
    kernel = Kernel(components(a * x, b * x), (x,), (a, b))
    block = Block(
      kernel, np.array([[0], [0]]), np.array([[BIG, 1.0], [-BIG, 0.0]]),
      np.array([[0], [0]]), np.ones((2, 1)),
    )
    result = assemble([block], np.array([1.0]), 1)
    self.assertEqual(result.residual[0], 1.0)
    self.assertEqual(result.jacobian[0, 0], 1.0)
    # Both controls lose the unit even when the final sum is compensated.
    subtotals = [math.fsum((BIG, 1.0)), math.fsum((-BIG, 0.0))]
    self.assertEqual(math.fsum(subtotals), 0.0)
    collapsed = coo_matrix(
      ([BIG, 1.0, -BIG], ([0, 0, 0], [0, 0, 0])), shape=(1, 1),
    ).tocsr()
    self.assertEqual(collapsed[0, 0], 0.0)

  def test_repeated_local_dofs_and_signed_nonunit_incidence(self):
    x, y, z = symbols("x y z")
    kernel = Kernel(components(BIG * x, y, -BIG * z), (x, y, z))
    # Three independently differentiated local slots name one global unknown.
    block = one_entity(
      kernel, (0, 0, 0), rows=(0, 1, 2), incidence=(0.1, 0.5, -0.5),
    )
    result = assemble([block], np.array([1.0]), 3)
    expected = np.array([0.1, 0.5, -0.5])
    np.testing.assert_array_equal(result.residual, expected)
    np.testing.assert_array_equal(result.jacobian.toarray()[:, 0], expected)
    self.assertEqual(result.residual[1] + result.residual[2], 0.0)
    self.assertEqual(result.jacobian[1, 0] + result.jacobian[2, 0], 0.0)
    self.assertEqual(result.jacobian.nnz, 3)

  def test_incidence_product_roundoff_matches_binary64_fraction_oracle(self):
    x, coefficient = symbols("x coefficient")
    kernel = Kernel(coefficient * x, (x,), (coefficient,))
    coefficients = (BIG + 2.0, 1.0, -BIG)
    for order in permutations(range(3)):
      selected = np.array([coefficients[i] for i in order])
      for weight in (0.1, 0.5, -0.5):
        with self.subTest(order=order, weight=weight):
          block = Block(
            kernel, np.zeros((3, 1), dtype=int), selected[:, None],
            np.zeros((3, 1), dtype=int), np.full((3, 1), weight),
          )
          expected = exact_weighted_sum(selected, [weight] * 3)
          result = assemble([block], np.array([1.0]), 1)
          self.assertEqual(result.residual[0], expected)
          self.assertEqual(result.jacobian[0, 0], expected)
    # Final fsum alone cannot repair already rounded incidence products.
    ordinary_products = math.fsum(value * 0.1 for value in coefficients)
    exact_products = exact_weighted_sum(coefficients, [0.1] * 3)
    self.assertEqual(ordinary_products, 0.35)
    self.assertEqual(exact_products, 0.30000000000000004)
    self.assertNotEqual(ordinary_products, exact_products)

  def test_duplicate_incidence_rows_preserve_every_weighted_component(self):
    x, = symbols("x")
    coefficients = (BIG + 2.0, 1.0, -BIG)
    kernel = Kernel(components(*(value * x for value in coefficients)), (x,))
    block = one_entity(kernel, (0,), rows=(0, 0), incidence=(0.1, -0.5))
    expected = exact_weighted_sum(
      list(coefficients) * 2, [0.1] * 3 + [-0.5] * 3,
    )
    result = assemble([block], np.array([1.0]), 1)
    self.assertEqual(result.residual[0], expected)
    self.assertEqual(result.jacobian[0, 0], expected)
    self.assertEqual(result.jacobian.nnz, 1)

  def test_explicit_fma_preserves_the_stored_coefficient_affine_law(self):
    x, g, b = symbols("x g b")
    conductance, pressure_square = 1e-12, 9e10
    intercept = float(conductance * pressure_square)
    state = np.array([pressure_square, conductance, intercept])
    expected = float(
      Fraction.from_float(intercept)
      - Fraction.from_float(conductance) * Fraction.from_float(pressure_square)
    )
    kernel = Kernel(fma(-g, x, b), (x, g, b))
    result = assemble([one_entity(kernel, (0, 1, 2))], state, 1)
    self.assertEqual(result.residual[0], expected)
    self.assertNotEqual(expected, 0.0)
    np.testing.assert_array_equal(
      result.jacobian.toarray()[0], [-conductance, -pressure_square, 1.0],
    )
    evaluate_source = kernel.source.split("void multiply")[0]
    self.assertIn("fma(", evaluate_source)
    self.assertIn("-ffp-contract=off", kernel.compile_command)
    # These are different stored-coefficient computations, despite an algebraic
    # equivalence one might infer by forgetting the rounding of the intercept.
    ordinary = Kernel(b - g * x, (x, g, b))
    reservoir = Kernel(g * (x - x), (x, g, b))
    for control in (ordinary, reservoir):
      values, _ = control.evaluate([state])
      self.assertEqual(values[0, 0], 0.0)
      self.assertNotEqual(values[0, 0], expected)

  def test_generated_derivatives_against_independent_calculus_and_differences(self):
    x, y = symbols("x y")
    expression = x * x / (y + 2) + sqrt(x + y + 4) - x * y
    kernel = Kernel(expression, (x, y), name="independent_calculus")
    state = np.array([1.25, 0.75])
    result = assemble([one_entity(kernel, (0, 1))], state, 1)
    u, v = state
    root = math.sqrt(u + v + 4)
    expected_value = u * u / (v + 2) + root - u * v
    expected_derivative = np.array([
      2 * u / (v + 2) + 0.5 / root - v,
      -u * u / (v + 2)**2 + 0.5 / root - u,
    ])
    self.assertAlmostEqual(result.residual[0], expected_value, places=15)
    np.testing.assert_allclose(
      result.jacobian.toarray()[0], expected_derivative, rtol=3e-15, atol=3e-15,
    )
    step = 2e-6
    finite_difference = []
    for axis in np.eye(2):
      plus, _ = kernel.evaluate([state + step * axis])
      minus, _ = kernel.evaluate([state - step * axis])
      finite_difference.append((plus[0, 0] - minus[0, 0]) / (2 * step))
    np.testing.assert_allclose(
      result.jacobian.toarray()[0], finite_difference, rtol=2e-9, atol=2e-9,
    )
    self.assertIn("void evaluate", kernel.source)
    self.assertIn("derivatives[k *", kernel.source)
    self.assertIn("sqrt(", kernel.source)
    self.assertIn("-fno-fast-math", kernel.compile_command)
    self.assertIn("-ffp-contract=off", kernel.compile_command)
    # Changing the authored expression generates different executable code.
    cubic = Kernel(x**3, (x,))
    square = Kernel(x**2, (x,))
    self.assertNotEqual(cubic.source, square.source)
    self.assertNotEqual(cubic.digest, square.digest)
    for other, expected in ((cubic, (8.0, 12.0)), (square, (4.0, 4.0))):
      values, derivatives = other.evaluate([[2.0]])
      self.assertEqual(values[0, 0], expected[0])
      self.assertEqual(math.fsum(derivatives[0]), expected[1])


if __name__ == "__main__":
  unittest.main()
