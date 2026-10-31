# SPDX-License-Identifier: GPL-3.0-or-later
"""Retain local components through incidence and unique-coordinate reduction."""

from dataclasses import dataclass
import math

import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class Block:
  kernel: object
  variable_ids: np.ndarray
  parameters: np.ndarray
  rows: np.ndarray
  incidence: np.ndarray

  def __post_init__(self):
    self.variable_ids = np.asarray(self.variable_ids, dtype=np.int64)
    self.parameters = np.asarray(self.parameters, dtype=float)
    self.rows = np.asarray(self.rows, dtype=np.int64)
    self.incidence = np.asarray(self.incidence, dtype=float)
    count = len(self.variable_ids)
    if self.variable_ids.shape != (count, len(self.kernel.variables)):
      raise ValueError("Variable map shape does not match the kernel")
    if self.parameters.shape != (count, len(self.kernel.parameters)):
      raise ValueError("Parameter shape does not match the kernel")
    if self.rows.ndim != 2 or self.rows.shape != self.incidence.shape:
      raise ValueError("Incidence rows and weights must have equal matrix shapes")
    if len(self.rows) != count or np.any(self.variable_ids < 0):
      raise ValueError("Invalid local-to-global variable map")
    if not np.all(np.isfinite(self.incidence)):
      raise ValueError("Incidence weights must be finite")


@dataclass
class Assembly:
  residual: np.ndarray
  jacobian: csr_matrix
  value_components: int
  derivative_components: int


def assemble(blocks, state, nrows):
  """Round once per global residual row and per sparse derivative coordinate.

  No local derivative subtotal and no sparse duplicate summation is permitted.
  FMA remainders from incidence weighting enter the same component bins.
  """
  state = np.asarray(state, dtype=float)
  if state.ndim != 1 or not np.all(np.isfinite(state)) or nrows < 1:
    raise ValueError("Assembly needs a finite vector and a positive row count")
  residual_bins = [[] for _ in range(nrows)]
  derivative_bins = {}
  value_count = derivative_count = 0
  for block in blocks:
    if block.variable_ids.size and block.variable_ids.max() >= len(state):
      raise ValueError("Variable map exceeds the global state")
    if block.rows.size and block.rows.max() >= nrows:
      raise ValueError("Incidence row exceeds the residual")
    inputs = np.concatenate([state[block.variable_ids], block.parameters], axis=1)
    values, derivatives = block.kernel.evaluate(inputs)
    for slot in range(block.rows.shape[1]):
      rows = block.rows[:, slot]
      weights = block.incidence[:, slot, None]
      vh, vl = block.kernel.multiply_parts(values, weights)
      dh, dl = block.kernel.multiply_parts(derivatives, weights)
      for entity, row in enumerate(rows):
        if row < 0 or weights[entity, 0] == 0:
          continue
        residual_bins[row].extend(vh[entity])
        residual_bins[row].extend(vl[entity])
        value_count += 2 * values.shape[1]
        for part, local_column in enumerate(block.kernel.derivative_variables):
          column = int(block.variable_ids[entity, local_column])
          bucket = derivative_bins.setdefault((int(row), column), [])
          bucket.extend((dh[entity, part], dl[entity, part]))
          derivative_count += 2
  residual = np.array([math.fsum(parts) for parts in residual_bins])
  coordinates = sorted(derivative_bins)
  rows = [item[0] for item in coordinates]
  columns = [item[1] for item in coordinates]
  values = [math.fsum(derivative_bins[item]) for item in coordinates]
  jacobian = csr_matrix((values, (rows, columns)), shape=(nrows, len(state)))
  if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(jacobian.data)):
    raise FloatingPointError("Global reduction overflow")
  return Assembly(residual, jacobian, value_count, derivative_count)
