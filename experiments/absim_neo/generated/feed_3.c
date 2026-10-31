/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Generated from immutable local expressions; do not edit. */
#include <math.h>
#include <stddef.h>
void evaluate(size_t count, const double *inputs, double *values,
  double *derivatives) {
  for (size_t k = 0; k < count; ++k) {
    const double *in = inputs + k * 10;
    const double t0 = ((0x1.0000000000000p+1) * in[4]);
    const double t1 = (t0 * in[5]);
    const double t2 = (t1 * in[6]);
    const double t3 = (t2 * in[7]);
    const double t4 = (in[0] - in[3]);
    const double t5 = (t3 * t4);
    const double t6 = (in[1] - in[0]);
    const double t7 = (in[8] * t6);
    const double t8 = (t3 * t7);
    const double t9 = (in[2] - in[0]);
    const double t10 = (in[9] * t9);
    const double t11 = (t3 * t10);
    const double t12 = (t3 * (0x1.0000000000000p+0));
    const double t13 = (-(0x1.0000000000000p+0));
    const double t14 = (t3 * t13);
    const double t15 = (in[8] * t13);
    const double t16 = (t3 * t15);
    const double t17 = (in[8] * (0x1.0000000000000p+0));
    const double t18 = (t3 * t17);
    const double t19 = (in[9] * t13);
    const double t20 = (t3 * t19);
    const double t21 = (in[9] * (0x1.0000000000000p+0));
    const double t22 = (t3 * t21);
    values[k * 3 + 0] = t5;
    values[k * 3 + 1] = t8;
    values[k * 3 + 2] = t11;
    derivatives[k * 6 + 0] = t12;
    derivatives[k * 6 + 1] = t14;
    derivatives[k * 6 + 2] = t16;
    derivatives[k * 6 + 3] = t18;
    derivatives[k * 6 + 4] = t20;
    derivatives[k * 6 + 5] = t22;
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
