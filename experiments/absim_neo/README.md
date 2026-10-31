# absim neo: executable mathematical kernel experiment

This experiment implements an isothermal gas film coupled to a finite pressure
port in two native spatial representations. It generates and compiles C value
and derivative kernels from short mathematical expressions, then assembles
retained numerical components through explicit incidence.

The implementation is independent of the reference libraries: it imports
neither numerical runtime and preserves neither internal API. Everything added
is confined to this directory. The original pyfvm source, authorship metadata
and licence remain intact. [REFERENCE.md](REFERENCE.md) records the actual
pinned pyfvm study and the attribution.

## Reproduce

Requirements: Python 3.12 or later, NumPy, SciPy, and a C99 compiler with `fma`
and a maths library. The committed receipt used Python 3.12.14, NumPy 2.3.5,
SciPy 1.17.0 and GCC 13.3.0 on Linux x86-64. Other compiler/platform combinations
have not been qualified by this run. The compiler is selected by `CC`, default
`cc`; compilation uses `-O2 -fno-fast-math -ffp-contract=off`.

```bash
git clone --branch agent/absim-neo-poc --single-branch https://github.com/soraros/pyfvm.git
cd pyfvm/experiments/absim_neo
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run.py --output RESULTS.json --emit-kernels generated
```

`run.py` executes the tests, refinement study, partial-feed steady and transient
cases, independent error measurements and precision controls. It writes the
environment, source SHA-256 identities, line counts and measured results to
[RESULTS.json](RESULTS.json). The [generated](generated) directory contains the
physical kernels emitted during that run; the executable always generates them
from the authoring expressions, so these files are inspectable output rather
than a second maintained implementation. Temporary shared libraries are built
outside the checkout.

## Mathematical authoring and generation

The complete diffusion law is authored as:

```python
left, right, h, metric, p0, mu, rt = symbols("left right h metric p0 mu rt")
conductance = h**3 * p0 / (12 * mu * rt)
edge = Kernel(
  components(conductance * metric * (left - right)),
  variables=(left, right, h),
  parameters=(metric, p0, mu, rt),
  name="film_edge",
)
```

The geometry supplies `metric`; the field space supplies the native unknown
indices and incidence. The same form therefore supplies both discretisations.
There are no handwritten residual Jacobians in the production model.

`Expr` is an immutable ordered tree. `components(a, b, c)` declares a sum whose
additive parts remain separate until global reduction. Ordinary `a + b` records
one ordered floating-point addition. The language supports arithmetic, integer
powers, square root and explicit fused multiply-add. It does not canonicalise,
factor, expand or reorder an authored expression.

The compiler differentiates each discrete component by the ordinary chain and
product rules, retaining every additive derivative path. It emits C statements
for values and those derivative components, compiles a batched shared library,
and calls it through `ctypes`. This is executable native code generation;
derivatives are not finite differences or prewritten operator tables. Exact
identical expression nodes may share a generated temporary, without changing
operand order. The small differentiation routine is limited to this scalar
language; it is not a general AD framework or PDE compiler.

## Physical problem

All pressures are absolute when used as physical data. The square has side
length `L = 0.02 m`, constant gap `h = 8e-6 m`, a grounded right edge and three
sealed edges. Parameters are `p0 = 101325 Pa`, supply pressure `200000 Pa`,
viscosity `1.81e-5 Pa s`, and `RT = 287.05 * 293.15 m2/s2`.

The native coordinate and pressure increment are

\[
s=\frac{p^2-p_0^2}{2p_0},\qquad
\Delta p(s)=\frac{2s}{\sqrt{1+2s/p_0}+1}.
\]

The native `s` coordinate has units Pa. Field, port, residuals, tangents and
observations retain it. Adding `p0` occurs only in the explicitly requested
`publish` operation.

The film equation, distributed feed and upstream mass flow are

\[
-\nabla\cdot(C\nabla s)+\beta(x)(s-s_P)+\partial_t\frac{hp}{RT}=0,
\quad C=\frac{h^3p_0}{12\mu RT},
\quad \beta(x)=2p_0g\,m(x),
\]

\[
\dot m_s=\gamma(s_s-s_P),\quad \gamma=2p_0G,\qquad
\int_\Omega\beta(s_P-s)\,dA+\partial_t\frac{V p_P}{RT}=\dot m_s.
\]

Here `g = 1e-13 kg/(s m2 Pa2)`, `G = 4e-17 kg/(s Pa2)`, and
`V = 1e-8 m3`. These are prescribed squared-pressure conductances; no choking or
resolved porous-body model is hidden inside them. The mask is either uniform or
the upper-left quadrant `x < L/2, y > L/2`. Even meshes align the discontinuous
patch with cells and triangles. The latter case has variation in both spatial
directions.

Backward Euler uses the physical inventory difference `h*p - h_old*p_old`.
Its stable decomposition is

\[
p_0(h-h_o)+(h-h_o)\Delta p(s_o)
+h\frac{2(s-s_o)}{\sqrt{1+2s/p_0}+\sqrt{1+2s_o/p_0}}.
\]

For P1, the numerator is formed from the native nodal changes first. If
`delta_i = s_i - old_i`, its components are `delta_0`,
`N1*(delta_1-delta_0)` and `N2*(delta_2-delta_0)`. They remain separate through
the storage law and assembly. Interpolating two large states and subtracting
their rounded values would erase useful changes before compensation begins.

The previous state, previous gap, origin, supply and geometry are held fixed
when differentiating the current residual with respect to gap.

## Two native representations

| Responsibility | CellFV | VertexP1 |
| --- | --- | --- |
| Unknowns | One `s` per rectangular cell | One `s` per triangle vertex |
| Diffusion geometry | Face length divided by centre distance; half-cell outlet distance | Actual triangle gradients, `-area * grad(Ni) dot grad(Nj)` per triangle pair |
| Interpolation | Cellwise constant | Native affine P1 interpolant |
| Feed | Cell midpoint integral | Consistent P1 source with degree-two triangle cubature |
| Nonlinear storage and load | Cell midpoint | Evaluate pressure at three triangle quadrature points, then integrate |
| Outlet | Explicit boundary face flux | Reaction of the constrained right-boundary nodal equations |
| Conservation demonstrated | Rectangular cell balances and global ledger | Paired algebraic exchanges, nodal equations and global ledger |

P1 values are never collapsed into a cellwise representation. In particular,
load is the declared quadrature of `sqrt(p0**2 + 2*p0*s) - p0`, rather than a
square root taken after averaging the squared-pressure coordinate.

For P1 this is a Galerkin formulation. Its nodal source integrals are not exact
geometric median-dual-volume integrals. The experiment does not reconstruct an
equilibrated physical flux field or prove separate primal-triangle balances.
Degree-two cubature is exact for the polynomial feed matrix in exact arithmetic;
it approximates the non-polynomial pressure and storage integrals.

## Arithmetic contract and its endpoint

Each generated kernel returns value components and derivative components.
Assembly applies incidence to each component separately, retaining both the
rounded product and its explicit FMA remainder. It groups all components by
global residual row or by sparse `(row, column)`, including repeated local DOFs
and contributions from different entities. `math.fsum` performs the final
reduction. Only unique coordinates enter SciPy CSR construction, so COO duplicate
reduction cannot destroy a component first.

The contract covers sums of the **computed local components** and their
incidence products. It does not recover information already discarded inside
an ordinary authored expression. `fma(a, b, c)` is explicit when a product plus
addition must have one rounding. Incidence compensation assumes no overflow or
underflow in the product/remainder. This is not arbitrary-precision arithmetic.

The endpoint matters: after a residual row or Jacobian entry has been rounded,
subsequent sparse matrix products, factorisation and solves use ordinary
binary64 arithmetic. An exact common-state primal flux does not imply bitwise
zero from every later sparse `J @ ones` computation. Generated derivatives are
derivatives of the authored real expression, not of the discontinuous mapping
between floating-point bit patterns.

## Correctness evidence

All 22 executable tests passed in the recorded run. They have three distinct
sources of evidence:

1. **Independent mathematical checks.** A second implementation constructs FV
   matrices directly and P1 gradient/consistent-mass matrices from coordinates,
   without production edges, weights, quadrature arrays or kernels. It uses a
   scalar-port Schur solution and analytic gap sensitivity. Nonlinear storage
   values are also checked against 70-digit Decimal evaluation of the physical
   `h*p` difference, with independently handwritten derivatives.
2. **Consistency and conservation checks.** Generated Jacobian actions and gap
   tangents are checked by finite differences and complete perturbed solves.
   Local incidence cancels internal exchanges, solved rows satisfy their
   residuals, and the external supply/storage/outlet ledger closes.
3. **Precision adversaries and negative controls.** Different entities and
   repeated local DOFs contribute `2**53`, `1`, `-2**53`; signed and non-unit
   incidence includes multiplication remainders. Early subtotals and ordinary
   sparse duplicates are retained as failing controls. A tiny native P1 time
   increment is checked at every storage row, because the global sum can hide
   a local interpolation error.

In the one-ULP native-history stress case, premature interpolation gives a
`33.3%` error in the storage rows while its total storage still agrees. The
retained temporal components reduce the native-row error to about `5.4e-16`
relative to the independent Decimal reference. The large common coordinate in
that control is an arithmetic stress input, not a claimed operating pressure.

The independently assembled full matrix reference uses binary64, not an
arbitrary-precision full solve. The Decimal reference is specifically for
physical storage integrands. The global observation ledger uses the same local
laws as the solver and is therefore a consistency check; it is not itself an
independent physical oracle.

For uniform feed there is also a continuum solution. With
`k = sqrt(beta/C)` and `Lambda = C*L*k*tanh(k*L)`,

\[
s_P=\frac{\gamma s_s}{\gamma+\Lambda},\qquad
s(x)=s_P\left(1-\frac{\cosh(kx)}{\cosh(kL)}\right),\qquad
\dot m=\Lambda s_P.
\]

Recorded relative errors in total upstream flow are:

| Subdivisions per side | CellFV | VertexP1 |
| ---: | ---: | ---: |
| 4 | `1.510e-2` | `5.811e-3` |
| 8 | `3.844e-3` | `1.467e-3` |
| 16 | `9.655e-4` | `3.679e-4` |

The reduction by approximately four on each refinement supports second-order
convergence for this problem. It does not establish general mesh/error bounds.
Independent steady and transient state/Jacobian/tangent errors are around
`1e-15` or smaller in the recorded small cases; their exact definitions and
values are in `RESULTS.json`.

The component adversary returns value and derivative `1`; early local subtotal
followed by `fsum` returns `0`. With non-unit incidence, one control gives `0.35`
after rounding products first, whereas retaining product remainders gives the
exact stored-input rational result `0.30000000000000004`.

A port increment of `2**-40 Pa = 9.094947017729282e-13 Pa` survives observation.
Restoring an absolute binary64 pressure at `101325 Pa` rounds that increment
away. Conservation is claimed for the native centred state; the rounded
published pressure field does not certify preservation of that increment.

## Implementation responsibility and cost

All human-written Python files, including tests and the execution driver, are
counted in `RESULTS.json`; blank lines and docstrings are included. Generated
physical C lines are counted separately. The delivered implementation contains
882 core Python lines, 678 test lines and 253 reproduction-driver lines: 1,813
human-written Python lines in total. Eleven generated physical C kernels add
615 lines. Documentation and requirements are additional. The small authoring snippet above is
not presented as the cost of the whole system.

| File | Lines | Responsibility |
| --- | ---: | --- |
| `neo/forms.py` | 146 | Ordered immutable scalar language, explicit components, symbolic derivative paths |
| `neo/kernels.py` | 154 | C emission, compiler invocation/cache, batched FFI, incidence-product remainders |
| `neo/assembly.py` | 88 | Native DOF maps, signed incidence, component grouping, unique CSR construction |
| `neo/spaces.py` | 155 | Both meshes, geometry, native interpolation and quadrature |
| `neo/bearing.py` | 337 | Constitutive forms, field/port coupling, boundary conditions, BE history data, Newton solve, centred observations, implicit sensitivity and publication |
| `neo/__init__.py` | 2 | Package identity and licence |
| `tests/` | 678 | Independent references, finite differences, physical/domain and precision controls |
| `run.py` | 253 | Execution receipt, refinement/partial-feed cases, source identities and kernel export |

External responsibilities remain explicit: NumPy supplies arrays, SciPy supplies
sparse storage and direct solves, the C compiler supplies machine code, and the
standard library supplies accurate final sums. Their implementation cost is not
included in the authored-line count. The original reference runtimes are not
dependencies.

The receipt includes full run time and single measurements of solve, observation
and tangent work. The latter exclude model construction/compilation, and the
test run may already have warmed the kernel cache. Python grouping and sparse
construction remain substantial work. There is no claimed performance gain
over either reference library.

## Acceptance and limits

Newton requires maximum residual at most `1e-12` times the declared upstream
mass-flow scale, preserves positive pressure and uses backtracking. The implicit
gap solve has a relative backward-residual gate of `1e-11`, with no absolute
floor. This ordinary sparse residual check is not a certified roundoff bound or
a forward-error theorem. Static stiffness is `-dW/dh`. At finite time step the
same derivative is a one-step load slope with history fixed, not dynamic
impedance.

This is an executable experiment, with the following work still outside scope:

- Arbitrary unstructured cell FV, hybrid cell/face or mixed flux spaces.
- Equilibrated P1 flux reconstruction and output-oriented error control.
- Disconnected floating components and automatic pressure-origin selection.
- General boundary/source/constitutive languages, choking and resolved porous depth.
- BDF histories, exact restart, adaptive transient integration and moving geometry.
- Geometry derivatives, higher derivatives, adjoints and certified linear actions.
- Underflow/overflow robustness, globally compensated linear algebra and large-scale performance qualification.

The PoC demonstrates that generated mathematical kernels, two native spaces and
explicit arithmetic obligations fit in a small direct implementation. Extending
its physical or numerical scope requires implementing the corresponding
responsibility; a new wrapper or common array shape cannot supply it.
