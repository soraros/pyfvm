/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Generated from immutable local expressions; do not edit. */
#include <math.h>
#include <stddef.h>
void evaluate(size_t count, const double *inputs, double *values,
  double *derivatives) {
  for (size_t k = 0; k < count; ++k) {
    const double *in = inputs + k * 11;
    const double t0 = (in[2] * in[3]);
    const double t1 = (in[5] * in[6]);
    const double t2 = (t0 / t1);
    const double t3 = (t2 * in[4]);
    const double t4 = (in[1] - in[7]);
    const double t5 = (t3 * t4);
    const double t6 = (t2 * t4);
    const double t7 = ((0x1.0000000000000p+1) * in[10]);
    const double t8 = (t7 / in[4]);
    const double t9 = ((0x1.0000000000000p+0) + t8);
    const double t10 = sqrt(t9);
    const double t11 = (t10 + (0x1.0000000000000p+0));
    const double t12 = (t7 / t11);
    const double t13 = (t6 * t12);
    const double t14 = (t2 * in[1]);
    const double t15 = (t14 * (0x1.0000000000000p+1));
    const double t16 = ((0x1.0000000000000p+1) * in[0]);
    const double t17 = (t16 / in[4]);
    const double t18 = ((0x1.0000000000000p+0) + t17);
    const double t19 = sqrt(t18);
    const double t20 = (t19 + t10);
    const double t21 = (t15 / t20);
    const double t22 = (in[0] - in[10]);
    const double t23 = (t21 * t22);
    const double t24 = (t3 * (0x1.0000000000000p+0));
    const double t25 = (t2 * (0x1.0000000000000p+0));
    const double t26 = (t25 * t12);
    const double t27 = ((0x1.0000000000000p+1) * (0x1.0000000000000p+0));
    const double t28 = (t27 / in[4]);
    const double t29 = ((0x1.0000000000000p+1) * t19);
    const double t30 = (t28 / t29);
    const double t31 = (t15 * t30);
    const double t32 = (-t31);
    const double t33 = (t20 * t20);
    const double t34 = (t32 / t33);
    const double t35 = (t34 * t22);
    const double t36 = (t21 * (0x1.0000000000000p+0));
    const double t37 = (t25 * (0x1.0000000000000p+1));
    const double t38 = (t37 / t20);
    const double t39 = (t38 * t22);
    values[k * 3 + 0] = t5;
    values[k * 3 + 1] = t13;
    values[k * 3 + 2] = t23;
    derivatives[k * 5 + 0] = t24;
    derivatives[k * 5 + 1] = t26;
    derivatives[k * 5 + 2] = t35;
    derivatives[k * 5 + 3] = t36;
    derivatives[k * 5 + 4] = t39;
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
