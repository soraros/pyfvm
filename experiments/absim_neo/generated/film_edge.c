/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Generated from immutable local expressions; do not edit. */
#include <math.h>
#include <stddef.h>
void evaluate(size_t count, const double *inputs, double *values,
  double *derivatives) {
  for (size_t k = 0; k < count; ++k) {
    const double *in = inputs + k * 7;
    const double t0 = (in[2] * in[2] * in[2]);
    const double t1 = (t0 * in[4]);
    const double t2 = ((0x1.8000000000000p+3) * in[5]);
    const double t3 = (t2 * in[6]);
    const double t4 = (t1 / t3);
    const double t5 = (t4 * in[3]);
    const double t6 = (in[0] - in[1]);
    const double t7 = (t5 * t6);
    const double t8 = (t5 * (0x1.0000000000000p+0));
    const double t9 = (-(0x1.0000000000000p+0));
    const double t10 = (t5 * t9);
    const double t11 = (in[2] * in[2]);
    const double t12 = ((0x1.8000000000000p+1) * t11);
    const double t13 = (t12 * (0x1.0000000000000p+0));
    const double t14 = (t13 * in[4]);
    const double t15 = (t14 / t3);
    const double t16 = (t15 * in[3]);
    const double t17 = (t16 * t6);
    values[k * 1 + 0] = t7;
    derivatives[k * 3 + 0] = t8;
    derivatives[k * 3 + 1] = t10;
    derivatives[k * 3 + 2] = t17;
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
