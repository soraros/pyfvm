/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Generated from immutable local expressions; do not edit. */
#include <math.h>
#include <stddef.h>
void evaluate(size_t count, const double *inputs, double *values,
  double *derivatives) {
  for (size_t k = 0; k < count; ++k) {
    const double *in = inputs + k * 2;
    const double t0 = ((0x1.0000000000000p+1) * in[0]);
    const double t1 = (t0 / in[1]);
    const double t2 = ((0x1.0000000000000p+0) + t1);
    const double t3 = sqrt(t2);
    const double t4 = (t3 + (0x1.0000000000000p+0));
    const double t5 = (t0 / t4);
    const double t6 = ((0x1.0000000000000p+1) * (0x1.0000000000000p+0));
    const double t7 = (t6 / t4);
    const double t8 = (t6 / in[1]);
    const double t9 = ((0x1.0000000000000p+1) * t3);
    const double t10 = (t8 / t9);
    const double t11 = (t0 * t10);
    const double t12 = (-t11);
    const double t13 = (t4 * t4);
    const double t14 = (t12 / t13);
    values[k * 1 + 0] = t5;
    derivatives[k * 2 + 0] = t7;
    derivatives[k * 2 + 1] = t14;
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
