/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Generated from immutable local expressions; do not edit. */
#include <math.h>
#include <stddef.h>
void evaluate(size_t count, const double *inputs, double *values,
  double *derivatives) {
  for (size_t k = 0; k < count; ++k) {
    const double *in = inputs + k * 8;
    const double t0 = ((0x1.0000000000000p+1) * in[2]);
    const double t1 = (t0 * in[3]);
    const double t2 = (t1 * in[4]);
    const double t3 = (t2 * in[5]);
    const double t4 = (in[0] - in[1]);
    const double t5 = (t3 * t4);
    const double t6 = (t3 * (0x1.0000000000000p+0));
    const double t7 = (-(0x1.0000000000000p+0));
    const double t8 = (t3 * t7);
    values[k * 1 + 0] = t5;
    derivatives[k * 2 + 0] = t6;
    derivatives[k * 2 + 1] = t8;
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
