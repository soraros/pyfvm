/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Generated from immutable local expressions; do not edit. */
#include <math.h>
#include <stddef.h>
void evaluate(size_t count, const double *inputs, double *values,
  double *derivatives) {
  for (size_t k = 0; k < count; ++k) {
    const double *in = inputs + k * 7;
    const double t0 = (in[3] * (0x1.0000000000000p+1));
    const double t1 = (in[1] - in[0]);
    const double t2 = (in[5] * t1);
    const double t3 = (in[0] + t2);
    const double t4 = (in[2] - in[0]);
    const double t5 = (in[6] * t4);
    const double t6 = (t3 + t5);
    const double t7 = (t0 * t6);
    const double t8 = ((0x1.0000000000000p+1) * t6);
    const double t9 = (t8 / in[4]);
    const double t10 = ((0x1.0000000000000p+0) + t9);
    const double t11 = sqrt(t10);
    const double t12 = (t11 + (0x1.0000000000000p+0));
    const double t13 = (t7 / t12);
    const double t14 = (t0 * (0x1.0000000000000p+0));
    const double t15 = (t14 / t12);
    const double t16 = (-(0x1.0000000000000p+0));
    const double t17 = (in[5] * t16);
    const double t18 = (t0 * t17);
    const double t19 = (t18 / t12);
    const double t20 = (in[6] * t16);
    const double t21 = (t0 * t20);
    const double t22 = (t21 / t12);
    const double t23 = ((0x1.0000000000000p+1) * (0x1.0000000000000p+0));
    const double t24 = (t23 / in[4]);
    const double t25 = ((0x1.0000000000000p+1) * t11);
    const double t26 = (t24 / t25);
    const double t27 = (t7 * t26);
    const double t28 = (-t27);
    const double t29 = (t12 * t12);
    const double t30 = (t28 / t29);
    const double t31 = ((0x1.0000000000000p+1) * t17);
    const double t32 = (t31 / in[4]);
    const double t33 = (t32 / t25);
    const double t34 = (t7 * t33);
    const double t35 = (-t34);
    const double t36 = (t35 / t29);
    const double t37 = ((0x1.0000000000000p+1) * t20);
    const double t38 = (t37 / in[4]);
    const double t39 = (t38 / t25);
    const double t40 = (t7 * t39);
    const double t41 = (-t40);
    const double t42 = (t41 / t29);
    const double t43 = (in[5] * (0x1.0000000000000p+0));
    const double t44 = (t0 * t43);
    const double t45 = (t44 / t12);
    const double t46 = ((0x1.0000000000000p+1) * t43);
    const double t47 = (t46 / in[4]);
    const double t48 = (t47 / t25);
    const double t49 = (t7 * t48);
    const double t50 = (-t49);
    const double t51 = (t50 / t29);
    const double t52 = (in[6] * (0x1.0000000000000p+0));
    const double t53 = (t0 * t52);
    const double t54 = (t53 / t12);
    const double t55 = ((0x1.0000000000000p+1) * t52);
    const double t56 = (t55 / in[4]);
    const double t57 = (t56 / t25);
    const double t58 = (t7 * t57);
    const double t59 = (-t58);
    const double t60 = (t59 / t29);
    values[k * 1 + 0] = t13;
    derivatives[k * 10 + 0] = t15;
    derivatives[k * 10 + 1] = t19;
    derivatives[k * 10 + 2] = t22;
    derivatives[k * 10 + 3] = t30;
    derivatives[k * 10 + 4] = t36;
    derivatives[k * 10 + 5] = t42;
    derivatives[k * 10 + 6] = t45;
    derivatives[k * 10 + 7] = t51;
    derivatives[k * 10 + 8] = t54;
    derivatives[k * 10 + 9] = t60;
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
