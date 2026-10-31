# SPDX-License-Identifier: GPL-3.0-or-later
"""Run the independent checks and record a reproducible numerical receipt."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
import unittest

import numpy as np
import scipy
from scipy.integrate import quad

from neo.assembly import Block, assemble
from neo.bearing import Bearing, Parameters, pressure_increment
from neo.forms import components, fma, symbols
from neo.kernels import Kernel
from neo.spaces import cell_fv, vertex_p1


ROOT = Path(__file__).resolve().parent


def relative_error(actual, expected):
  return float(np.max(np.abs(np.asarray(actual) - expected))
                / max(np.max(np.abs(expected)), np.finfo(float).tiny))


def oracle_module():
  specification = importlib.util.spec_from_file_location(
    "independent_oracles", ROOT / "tests/test_oracles.py",
  )
  module = importlib.util.module_from_spec(specification)
  specification.loader.exec_module(module)
  return module


def independent_errors(constructor, kind):
  oracle = oracle_module()
  n, length, p = 4, 0.01, Parameters()
  model = Bearing(constructor(n, length), p)
  state = model.solve()
  expected, tangent, matrix, flow, _ = oracle.steady_oracle(kind, n, length, p)
  actual = model.evaluate(state)
  free = np.setdiff1d(np.arange(model.nstate), model.fixed)
  result = {
    "steady_state_relative_max_error": relative_error(state, expected),
    "steady_jacobian_relative_max_error": relative_error(
      actual.jacobian.toarray()[free], matrix[free]),
    "steady_gap_tangent_relative_max_error": relative_error(
      model.gap_tangent(state)[0], tangent),
    "steady_flow_relative_error": abs(model.observe(state)["upstream_mass_kg_s"] / flow - 1),
  }
  feed = lambda xy: xy[0] < length / 2 and xy[1] > length / 2
  previous = oracle.steady_oracle(kind, n, length, p, feed_mask=feed)[0]
  gap, dt = 9e-6, 0.02
  transient = Bearing(constructor(n, length), p, dt=dt, previous=previous,
    previous_gap=p.gap, feed_mask=feed)
  state = transient.solve(h=gap)
  residual, matrix, direct, _ = oracle.transient_oracle(
    kind, n, length, p, state, previous, gap, p.gap, dt, feed_mask=feed,
  )
  actual = transient.evaluate(state, h=gap)
  exact_tangent = np.linalg.solve(matrix, -direct)
  direction = np.linspace(-0.6, 0.7, len(state)) * 1e4
  step = 2e-4
  finite = (transient.evaluate(state + step * direction, h=gap).residual
            - transient.evaluate(state - step * direction, h=gap).residual) / (2 * step)
  hstep = 1e-5 * gap
  finite_h = (transient.evaluate(state, h=gap + hstep).residual
              - transient.evaluate(state, h=gap - hstep).residual) / (2 * hstep)
  result.update({
    "be_independent_residual_max_kg_s": float(np.max(np.abs(residual))),
    "be_jacobian_relative_max_error": relative_error(
      actual.jacobian.toarray()[free], matrix[free]),
    "be_gap_tangent_relative_max_error": relative_error(
      transient.gap_tangent(state, h=gap)[0], exact_tangent),
    "be_state_direction_fd_relative_max_error": relative_error(finite[free], (matrix @ direction)[free]),
    "be_direct_gap_fd_relative_max_error": relative_error(finite_h, direct),
    "be_balance_kg_s": transient.observe(state, h=gap)["balance_kg_s"],
  })
  return result


def precision_probe():
  x, = symbols("x")
  amplitude = 2.0**53
  kernel = Kernel(components(amplitude * x, x, -amplitude * x), (x,))
  block = Block(kernel, [[0]], np.empty((1, 0)), [[0]], [[1.0]])
  result = assemble([block], [1.0], 1)
  naive = 0.0
  for value in (amplitude, 1.0, -amplitude):
    naive += value
  g, intercept = 1e-12, float(1e-12 * 9e10)
  affine = Kernel(components(fma(-g, x, intercept)), (x,))
  affine_value = float(affine.evaluate([[9e10]])[0][0, 0])
  model = Bearing(cell_fv(2, 0.02))
  state = np.zeros(model.nstate)
  state[-1] = 2.0**-40
  centred = model.observe(state)
  published = model.publish(state)
  return {
    "component_value": float(result.residual[0]),
    "component_derivative": float(result.jacobian[0, 0]),
    "ordinary_ordered_sum": naive,
    "fsum_after_local_subtotal": math.fsum([amplitude + 1.0, -amplitude]),
    "single_rounding_affine_value": affine_value,
    "ordinary_affine_value": intercept - g * 9e10,
    "centred_port_increment_Pa": centred["port_increment_Pa"],
    "published_port_increment_Pa": published["port_pressure_Pa"] - model.parameters.ambient,
    "centred_feed_kg_s": centred["film_feed_mass_kg_s"],
  }


def continuum_load(parameters, length):
  p = parameters
  c = p.gap**3 * p.ambient / (12 * p.viscosity * p.gas_rt)
  k = math.sqrt(2 * p.ambient * p.porous / c)
  admittance = c * length * k * math.tanh(k * length)
  gamma = 2 * p.ambient * p.upstream
  port = gamma * p.supply_coordinate / (gamma + admittance)
  value, error = quad(lambda x: length * float(pressure_increment(
    port * (1 - math.cosh(k * x) / math.cosh(k * length)), p.ambient,
  )), 0, length, epsabs=1e-12, epsrel=1e-12)
  return value, error


def main():
  run_started = time.perf_counter()
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, default=Path("RESULTS.json"))
  parser.add_argument("--emit-kernels", type=Path)
  args = parser.parse_args()
  suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
  stream = io.StringIO()
  started = time.perf_counter()
  tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
  test_seconds = time.perf_counter() - started
  print(stream.getvalue(), end="")
  if not tests.wasSuccessful():
    raise SystemExit(1)
  p, length = Parameters(), 0.02
  exact_load, quadrature_error = continuum_load(p, length)
  convergence, physical_cases, unique_kernels = [], [], {}
  for kind, constructor in (("cell", cell_fv), ("p1", vertex_p1)):
    for n in (4, 8, 16):
      model = Bearing(constructor(n, length), p)
      started = time.perf_counter()
      state = model.solve()
      observation = model.observe(state)
      _, load_tangent = model.gap_tangent(state)
      elapsed = time.perf_counter() - started
      exact_field, exact_port, exact_flow = model.exact_solution()
      convergence.append({
        "space": model.space.name, "n": n, "state_dofs": model.nstate,
        "native_sample_relative_max_error": relative_error(state[:-1], exact_field),
        "port_relative_error": abs(state[-1] / exact_port - 1),
        "flow_relative_error": abs(observation["upstream_mass_kg_s"] / exact_flow - 1),
        "load_relative_error": abs(observation["load_N"] / exact_load - 1),
        "load_N": observation["load_N"], "stiffness_N_m": -load_tangent,
        "mass_balance_kg_s": observation["balance_kg_s"],
        "solve_observe_tangent_seconds": elapsed,
      })
      for kernel in model.kernels.values():
        unique_kernels[kernel.digest] = kernel
    feed = lambda xy: xy[0] < length / 2 and xy[1] > length / 2
    steady = Bearing(constructor(8, length), p, feed_mask=feed)
    previous = steady.solve()
    transient = Bearing(constructor(8, length), p, dt=0.02, previous=previous,
      previous_gap=p.gap, feed_mask=feed)
    for label, model, h, state in (
      ("partial_steady", steady, p.gap, previous),
      ("partial_be_step", transient, 9e-6, transient.solve(h=9e-6)),
    ):
      observation = model.observe(state, h=h)
      _, derivative = model.gap_tangent(state, h=h)
      physical_cases.append({
        "case": label, "space": model.space.name, "n": 8,
        "state_dofs": model.nstate, "gap_m": h,
        "newton_iterations": model.iterations, "negative_gap_load_slope_N_m": -derivative,
        "slope_kind": "static_stiffness" if label == "partial_steady" else "backward_euler_step",
        **observation,
      })
  kernel_records = []
  if args.emit_kernels:
    args.emit_kernels.mkdir(parents=True, exist_ok=True)
  for digest, kernel in sorted(unique_kernels.items()):
    filename = f"{kernel.name}.c"
    if args.emit_kernels:
      (args.emit_kernels / filename).write_text(kernel.source)
    kernel_records.append({
      "name": kernel.name, "file": filename, "sha256": digest,
      "source_lines": len(kernel.source.splitlines()),
      "value_components": kernel.nvalue,
      "derivative_components": kernel.nderivative,
    })
  source_files = sorted(ROOT.glob("neo/*.py")) + sorted(ROOT.glob("tests/*.py")) + [Path(__file__)]
  costs = {}
  hashes = {}
  for path in source_files:
    relative = path.relative_to(ROOT).as_posix()
    costs[relative] = len(path.read_text().splitlines())
    hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
  compiler = subprocess.run(shlex.split(os.environ.get("CC", "cc")) + ["--version"],
    capture_output=True, text=True)
  receipt = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "public_reference": "soraros/pyfvm@cbf9d2bf2bdcfc2b6cc37ad46ed66b986cd8efac",
    "environment": {"python": platform.python_version(), "platform": platform.platform(),
      "numpy": np.__version__, "scipy": scipy.__version__,
      "compiler": compiler.stdout.splitlines()[0],
      "flags": ["-std=c99", "-O2", "-fPIC", "-shared", "-fno-fast-math", "-ffp-contract=off"]},
    "tests": {"run": tests.testsRun, "failures": len(tests.failures),
      "errors": len(tests.errors), "seconds": test_seconds,
      "command": "python -m unittest discover -s tests -v"},
    "precision": precision_probe(),
    "independent_errors": {kind: independent_errors(constructor, kind)
      for kind, constructor in (("cell", cell_fv), ("p1", vertex_p1))},
    "uniform_continuum_load_N": exact_load,
    "continuum_load_quadrature_error_estimate_N": quadrature_error,
    "convergence": convergence, "physical_cases": physical_cases,
    "source_lines": costs, "source_sha256": hashes,
    "generated_kernels": kernel_records,
    "generated_source_lines": sum(row["source_lines"] for row in kernel_records),
    "limitations": [
      "Independent complete matrix solves use binary64; only storage integrands use Decimal70.",
      "P1 conservation is nodal algebraic conservation and a Dirichlet-reaction global ledger.",
      "P1 nonlinear storage/load use degree-two cubature, not exact non-polynomial integration.",
      "Compensation ends at each residual row and sparse entry; later sparse algebra is ordinary float64.",
      "Incidence product compensation excludes overflow/underflow and does not undo rounding inside local expressions.",
      "Timings are single-process measurements, not a performance comparison with either reference library.",
    ],
  }
  receipt["complete_run_seconds_before_write"] = time.perf_counter() - run_started
  args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
  print(f"Wrote {args.output}: {tests.testsRun} tests passed; {len(kernel_records)} physical C kernels.")
  for row in convergence:
    print(f"{row['space']:9} n={row['n']:2} port error={row['port_relative_error']:.3e} "
          f"flow error={row['flow_relative_error']:.3e} load error={row['load_relative_error']:.3e}")


if __name__ == "__main__":
  main()
