/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Generated from immutable local expressions; do not edit. */
#include <math.h>
#include <stddef.h>
void evaluate(size_t count, const double *inputs, double *values,
  double *derivatives) {
  for (size_t k = 0; k < count; ++k) {
    const double *in = inputs + k * 4;
    const double t0 = ((0x1.0000000000000p+1) * in[1]);
    const double t1 = (t0 * in[2]);
    const double t2 = (in[0] - in[3]);
    const double t3 = (t1 * t2);
    const double t4 = (t1 * (0x1.0000000000000p+0));
    values[k * 1 + 0] = t3;
    derivatives[k * 1 + 0] = t4;
  }
}
void multiply(size_t count, const double *a, const double *b,
  double *high, double *low) {
  for (size_t k = 0; k < count; ++k) {
    const double p = a[k] * b[k];
    high[k] = p;
    low[k] = fma(a[k], b[k], -p);
  }
}
