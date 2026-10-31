/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Generated from immutable local expressions; do not edit. */
#include <math.h>
#include <stddef.h>
void evaluate(size_t count, const double *inputs, double *values,
  double *derivatives) {
  for (size_t k = 0; k < count; ++k) {
    const double *in = inputs + k * 5;
    const double t0 = (in[1] * (0x1.0000000000000p+1));
    const double t1 = (t0 * in[0]);
    const double t2 = ((0x1.0000000000000p+1) * in[0]);
    const double t3 = (t2 / in[2]);
    const double t4 = ((0x1.0000000000000p+0) + t3);
    const double t5 = sqrt(t4);
    const double t6 = (t5 + (0x1.0000000000000p+0));
    const double t7 = (t1 / t6);
    const double t8 = (t0 * (0x1.0000000000000p+0));
    const double t9 = (t8 / t6);
    const double t10 = ((0x1.0000000000000p+1) * (0x1.0000000000000p+0));
    const double t11 = (t10 / in[2]);
    const double t12 = ((0x1.0000000000000p+1) * t5);
    const double t13 = (t11 / t12);
    const double t14 = (t1 * t13);
    const double t15 = (-t14);
    const double t16 = (t6 * t6);
    const double t17 = (t15 / t16);
    values[k * 1 + 0] = t7;
    derivatives[k * 2 + 0] = t9;
    derivatives[k * 2 + 1] = t17;
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
