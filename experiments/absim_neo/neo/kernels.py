# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate and compile scalar value/tangent component kernels as C99 code."""

import ctypes
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

import numpy as np

from .forms import Expr, Form, components, derivative_parts


_CACHE = {}
_ARRAY = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")


class Kernel:
  """A generated batched kernel; local derivative columns stay decomposed."""

  def __init__(self, form, variables, parameters=(), name="local"):
    self.form = form if isinstance(form, Form) else components(form)
    self.variables = tuple(variables)
    self.parameters = tuple(parameters)
    self.name = name
    inputs = self.variables + self.parameters
    if len(set(inputs)) != len(inputs) or any(
      not isinstance(x, Expr) or x.op != "symbol" for x in inputs
    ):
      raise ValueError("Kernel inputs must be distinct symbols")
    self.ninput = len(inputs)
    self.nvalue = len(self.form.parts)
    tangent_parts, columns = [], []
    for part in self.form.parts:
      for column, variable in enumerate(self.variables):
        deriv = derivative_parts(part, variable)
        tangent_parts.extend(deriv)
        columns.extend([column] * len(deriv))
    self.derivative_variables = tuple(columns)
    self.nderivative = len(columns)
    names = {x: f"in[{i}]" for i, x in enumerate(inputs)}
    statements = []

    def emit(expression):
      if expression in names:
        return names[expression]
      if expression.op == "constant":
        return f"({float(expression.value).hex()})"
      if expression.op == "symbol":
        raise ValueError(f"Undeclared kernel symbol: {expression.value}")
      arguments = [emit(arg) for arg in expression.args]
      op = expression.op
      if op in ("add", "sub", "mul", "div"):
        token = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[op]
        body = f"({arguments[0]} {token} {arguments[1]})"
      elif op == "neg":
        body = f"(-{arguments[0]})"
      elif op == "sqrt":
        body = f"sqrt({arguments[0]})"
      elif op == "fma":
        body = f"fma({arguments[0]}, {arguments[1]}, {arguments[2]})"
      elif op == "pow":
        body = "(" + " * ".join([arguments[0]] * expression.value) + ")"
      else:
        raise ValueError(f"Unsupported operation: {op}")
      name = f"t{len(statements)}"
      statements.append(f"    const double {name} = {body};")
      names[expression] = name
      return name

    value_outputs = [emit(part) for part in self.form.parts]
    derivative_outputs = [emit(part) for part in tangent_parts]
    stores = [
      f"    values[k * {self.nvalue} + {j}] = {value};"
      for j, value in enumerate(value_outputs)
    ] + [
      f"    derivatives[k * {self.nderivative} + {j}] = {value};"
      for j, value in enumerate(derivative_outputs)
    ]
    self.source = "\n".join([
      "/* SPDX-License-Identifier: GPL-3.0-or-later */",
      "/* Generated from immutable local expressions; do not edit. */",
      "#include <math.h>",
      "#include <stddef.h>",
      "void evaluate(size_t count, const double *inputs, double *values,",
      "  double *derivatives) {",
      "  for (size_t k = 0; k < count; ++k) {",
      f"    const double *in = inputs + k * {self.ninput};",
      *statements,
      *stores,
      "  }",
      "}",
      "void multiply(size_t count, const double *a, const double *b,",
      "  double *high, double *low) {",
      "  for (size_t k = 0; k < count; ++k) {",
      "    const double p = a[k] * b[k];",
      "    high[k] = p;",
      "    low[k] = fma(a[k], b[k], -p);",
      "  }",
      "}",
      "",
    ])
    self.digest = hashlib.sha256(self.source.encode()).hexdigest()
    if self.digest not in _CACHE:
      directory = tempfile.TemporaryDirectory(prefix="absim-neo-")
      source_path = Path(directory.name) / "kernel.c"
      library_path = Path(directory.name) / "kernel.so"
      source_path.write_text(self.source)
      command = shlex.split(os.environ.get("CC", "cc")) + [
        "-std=c99", "-O2", "-fPIC", "-shared", "-fno-fast-math",
        "-ffp-contract=off", str(source_path), "-lm", "-o", str(library_path),
      ]
      result = subprocess.run(command, capture_output=True, text=True)
      if result.returncode:
        directory.cleanup()
        raise RuntimeError(f"C compiler failed:\n{result.stderr}")
      library = ctypes.CDLL(str(library_path))
      library.evaluate.argtypes = [ctypes.c_size_t, _ARRAY, _ARRAY, _ARRAY]
      library.evaluate.restype = None
      library.multiply.argtypes = [ctypes.c_size_t, _ARRAY, _ARRAY, _ARRAY, _ARRAY]
      library.multiply.restype = None
      _CACHE[self.digest] = (directory, library, command)
    self._directory, self._library, self.compile_command = _CACHE[self.digest]

  def evaluate(self, inputs):
    inputs = np.ascontiguousarray(inputs, dtype=np.float64)
    if inputs.ndim != 2 or inputs.shape[1] != self.ninput:
      raise ValueError(f"Expected inputs with {self.ninput} columns")
    if not np.all(np.isfinite(inputs)):
      raise FloatingPointError("Non-finite kernel input")
    values = np.empty((len(inputs), self.nvalue), dtype=np.float64)
    derivatives = np.empty((len(inputs), self.nderivative), dtype=np.float64)
    self._library.evaluate(len(inputs), inputs, values, derivatives)
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(derivatives)):
      raise FloatingPointError("Kernel domain violation or arithmetic overflow")
    return values, derivatives

  def multiply_parts(self, a, b):
    """Retain incidence-product roundoff with an FMA remainder.

    The two-part identity is exact when the product/remainder do not overflow
    or underflow. Underflow is outside this PoC's compensation guarantee.
    """
    a, b = np.broadcast_arrays(a, b)
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    high, low = np.empty_like(a), np.empty_like(a)
    self._library.multiply(a.size, a, b, high, low)
    if not np.all(np.isfinite(high)) or not np.all(np.isfinite(low)):
      raise FloatingPointError("Incidence multiplication overflow")
    return high, low
