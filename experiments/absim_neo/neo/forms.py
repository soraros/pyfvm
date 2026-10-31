# SPDX-License-Identifier: GPL-3.0-or-later
"""Immutable ordered scalar expressions and explicitly retained additive parts.

An Expr records arithmetic order. A Form declares that its parts must remain
separate through differentiation and assembly. This is deliberately not a
symbolic simplifier, a general tensor language, or an arbitrary-precision type.
"""

from dataclasses import dataclass
import math
import re


@dataclass(frozen=True)
class Expr:
  op: str
  args: tuple = ()
  value: object = None

  def __add__(self, other):
    return Expr("add", (self, as_expr(other)))

  def __radd__(self, other):
    return as_expr(other) + self

  def __sub__(self, other):
    return Expr("sub", (self, as_expr(other)))

  def __rsub__(self, other):
    return as_expr(other) - self

  def __mul__(self, other):
    return Expr("mul", (self, as_expr(other)))

  def __rmul__(self, other):
    return as_expr(other) * self

  def __truediv__(self, other):
    return Expr("div", (self, as_expr(other)))

  def __rtruediv__(self, other):
    return as_expr(other) / self

  def __neg__(self):
    return Expr("neg", (self,))

  def __pow__(self, exponent):
    if not isinstance(exponent, int) or not 0 <= exponent <= 8:
      raise ValueError("Only integer powers from zero to eight are supported")
    if exponent == 0:
      return as_expr(1)
    return Expr("pow", (self,), exponent)


def as_expr(value):
  if isinstance(value, Expr):
    return value
  value = float(value)
  if not math.isfinite(value):
    raise ValueError("Expression constants must be finite")
  return Expr("constant", value=value)


def symbols(names):
  names = names.split()
  if len(set(names)) != len(names) or any(
    re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name) is None for name in names
  ):
    raise ValueError("Symbols must have distinct identifier names")
  return tuple(Expr("symbol", value=name) for name in names)


def sqrt(expression):
  return Expr("sqrt", (as_expr(expression),))


def fma(a, b, c):
  """An explicitly authored, single-rounding product-plus-add operation."""
  return Expr("fma", tuple(as_expr(x) for x in (a, b, c)))


@dataclass(frozen=True)
class Form:
  parts: tuple

  def __add__(self, other):
    other = other if isinstance(other, Form) else components(other)
    return Form(self.parts + other.parts)

  def __sub__(self, other):
    other = other if isinstance(other, Form) else components(other)
    return self + (-other)

  def __neg__(self):
    return Form(tuple(-part for part in self.parts))

  def __mul__(self, factor):
    return Form(tuple(part * factor for part in self.parts))

  __rmul__ = __mul__


def components(*expressions):
  """Declare an additive sum whose parts round only at global reduction."""
  if not expressions:
    raise ValueError("A form needs at least one component")
  return Form(tuple(as_expr(expression) for expression in expressions))


def derivative_parts(expression, variable):
  """Differentiate the discrete expression, retaining every additive path.

  Product/chain rules produce streams, never a rounded local derivative sum.
  These are derivatives of the mathematical expression, not of the discontinuous
  mapping from floating-point bit patterns to floating-point bit patterns.
  """
  op, args = expression.op, expression.args
  if op == "constant":
    return ()
  if op == "symbol":
    return (as_expr(1),) if expression == variable else ()
  a = args[0]
  da = derivative_parts(a, variable)
  if op == "neg":
    return tuple(-part for part in da)
  if op == "sqrt":
    return tuple(part / (2 * sqrt(a)) for part in da)
  if op == "pow":
    n = expression.value
    return tuple((n * (a ** (n - 1))) * part for part in da)
  b = args[1]
  db = derivative_parts(b, variable)
  if op == "add":
    return da + db
  if op == "sub":
    return da + tuple(-part for part in db)
  if op == "mul":
    return tuple(part * b for part in da) + tuple(a * part for part in db)
  if op == "fma":
    return (tuple(part * b for part in da) + tuple(a * part for part in db)
            + derivative_parts(args[2], variable))
  if op == "div":
    return tuple(part / b for part in da) + tuple(
      -(a * part) / (b * b) for part in db
    )
  raise ValueError(f"Unsupported operation: {op}")
