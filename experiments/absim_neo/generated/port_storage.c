/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Generated from immutable local expressions; do not edit. */
#include <math.h>
#include <stddef.h>
void evaluate(size_t count, const double *inputs, double *values,
  double *derivatives) {
  for (size_t k = 0; k < count; ++k) {
    const double *in = inputs + k * 6;
    const double t0 = (in[2] * in[3]);
    const double t1 = (in[1] / t0);
    const double t2 = (t1 * (0x1.0000000000000p+1));
    const double t3 = (in[0] - in[5]);
    const double t4 = (t2 * t3);
    const double t5 = ((0x1.0000000000000p+1) * in[0]);
    const double t6 = (t5 / in[4]);
    const double t7 = ((0x1.0000000000000p+0) + t6);
    const double t8 = sqrt(t7);
    const double t9 = ((0x1.0000000000000p+1) * in[5]);
    const double t10 = (t9 / in[4]);
    const double t11 = ((0x1.0000000000000p+0) + t10);
    const double t12 = sqrt(t11);
    const double t13 = (t8 + t12);
    const double t14 = (t4 / t13);
    const double t15 = (t2 * (0x1.0000000000000p+0));
    const double t16 = (t15 / t13);
    const double t17 = ((0x1.0000000000000p+1) * (0x1.0000000000000p+0));
    const double t18 = (t17 / in[4]);
    const double t19 = ((0x1.0000000000000p+1) * t8);
    const double t20 = (t18 / t19);
    const double t21 = (t4 * t20);
    const double t22 = (-t21);
    const double t23 = (t13 * t13);
    const double t24 = (t22 / t23);
    values[k * 1 + 0] = t14;
    derivatives[k * 2 + 0] = t16;
    derivatives[k * 2 + 1] = t24;
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
