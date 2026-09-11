/* Tail rational approximation adapted from FreeBSD msun s_erf.c, also
 * used by the former libm dependency. The exponential is evaluated in bounded
 * vector batches by our C array core.
 *
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this software is freely
 * granted, provided that this notice is preserved.
 */
#ifndef QACCEL_ERFC_H
#define QACCEL_ERFC_H
static const double RA0 = -9.86494403484714822705e-03;
static const double RA1 = -6.93858572707181764372e-01;
static const double RA2 = -1.05586262253232909814e+01;
static const double RA3 = -6.23753324503260060396e+01;
static const double RA4 = -1.62396669462573470355e+02;
static const double RA5 = -1.84605092906711035994e+02;
static const double RA6 = -8.12874355063065934246e+01;
static const double RA7 = -9.81432934416914548592e+00;
static const double SA1 = 1.96512716674392571292e+01;
static const double SA2 = 1.37657754143519042600e+02;
static const double SA3 = 4.34565877475229228821e+02;
static const double SA4 = 6.45387271733267880336e+02;
static const double SA5 = 4.29008140027567833386e+02;
static const double SA6 = 1.08635005541779435134e+02;
static const double SA7 = 6.57024977031928170135e+00;
static const double SA8 = -6.04244152148580987438e-02;
static const double RB0 = -9.86494292470009928597e-03;
static const double RB1 = -7.99283237680523006574e-01;
static const double RB2 = -1.77579549177547519889e+01;
static const double RB3 = -1.60636384855821916062e+02;
static const double RB4 = -6.37566443368389627722e+02;
static const double RB5 = -1.02509513161107724954e+03;
static const double RB6 = -4.83519191608651397019e+02;
static const double SB1 = 3.03380607434824582924e+01;
static const double SB2 = 3.25792512996573918826e+02;
static const double SB3 = 1.53672958608443695994e+03;
static const double SB4 = 3.19985821950859553908e+03;
static const double SB5 = 2.55305040643316442583e+03;
static const double SB6 = 4.74528541206955367215e+02;
static const double SB7 = -2.24409524465858183362e+01;

static double erfc_tail_correction(double x, double *exponent) {
    double s = 1. / (x*x), r, den;
    if (x < 2.8571414947509765625) {
        r = RA0+s*(RA1+s*(RA2+s*(RA3+s*(RA4+s*(RA5+s*(RA6+s*RA7))))));
        den = 1.+s*(SA1+s*(SA2+s*(SA3+s*(SA4+s*(SA5+s*(SA6+s*(SA7+s*SA8)))))));
    } else {
        r = RB0+s*(RB1+s*(RB2+s*(RB3+s*(RB4+s*(RB5+s*RB6)))));
        den = 1.+s*(SB1+s*(SB2+s*(SB3+s*(SB4+s*(SB5+s*(SB6+s*SB7))))));
    }
    /* Split x so z*z is exact and retain the low part in the correction. */
    uint64_t bits;
    memcpy(&bits, &x, sizeof(bits));
    bits &= UINT64_C(0xffffffff00000000);
    double z;
    memcpy(&z, &bits, sizeof(z));
    *exponent = -z*z-.5625;
    return (z-x)*(z+x)+r/den;
}

static void erfc_array(const double *src, qintp stride, double *dst, qintp n) {
    enum { BLOCK = 256 };
    double main_exp[BLOCK], correction[BLOCK];
    for (qintp start=0; start<n; start+=BLOCK) {
        qintp count = n-start < BLOCK ? n-start : BLOCK;
        for (qintp j=0;j<count;j++) {
            double x = src[(start+j)*stride], ax = fabs(x);
            if (ax >= 1.25 && ax < 28. && x > -6.)
                correction[j] = erfc_tail_correction(ax, &main_exp[j]);
            else { main_exp[j]=0.; correction[j]=0.; }
        }
        qnp_exp_f64(main_exp,main_exp,count);
        qnp_exp_f64(correction,correction,count);
        for (qintp j=0;j<count;j++) {
            double x = src[(start+j)*stride], ax = fabs(x);
            if (ax >= 1.25 && ax < 28. && x > -6.) {
                double tail = main_exp[j]*correction[j]/ax;
                dst[start+j] = x < 0. ? 2.-tail : tail;
            } else dst[start+j] = erfc(x);
        }
    }
}
#endif
