'''
program:
created: 2016-06-02 -- 15 CEST
author: tc
'''

import time
import math
import cmath

import cython
import numpy
from scipy.special import erfc as erfc
from scipy.integrate import nquad as nquad


cdef double rad_sq(_v):
    cdef double tot = 0.0
    cdef int i
    for i in range(3):
        tot += _v[i] ** 2
    return tot


def out_vect(label, vector, _out):
    _out.write('%s: ' % label)
    for x in vector:
        _out.write('%+.12f ' % x)
    _out.write('\n')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef class HCorr:

    cdef double Ex, Ey, Ez, bx, by, bz, ex, ey, ez, b, e, bp, ep, beta_star
    cdef public double [:]A_vec
    cdef public double [:]B_vec
    cdef public double [:]E_vec
    cdef public double [:]b_vec
    cdef public double [:]e_vec
    cdef public double [:]bp_vec
    cdef public double [:]ep_vec
    cdef public double [:]two_ep_minus_bp_vec


    def __init__(self, A_vec, B_vec, E_vec, double beta_star_val):
        cdef int i
        self.A_vec = numpy.asarray(A_vec[:])
        self.B_vec = numpy.asarray(B_vec[:])
        self.E_vec = numpy.asarray(E_vec[:])
        self.beta_star = beta_star_val
        self.Ex, self.Ey, self.Ez = self.E_vec[:]
        self.b_vec = numpy.array([self.A_vec[i] - self.B_vec[i] for i in [0, 1, 2]])
        self.e_vec = numpy.array([self.A_vec[i] - self.E_vec[i] for i in [0, 1, 2]])
        self.bx, self.by, self.bz = self.b_vec[:]
        self.ex, self.ey, self.ez = self.e_vec[:]
        self.bp_vec = numpy.array([x / math.sqrt(2.0 * self.beta_star) for x in self.b_vec])
        self.ep_vec = numpy.array([x / math.sqrt(2.0 * self.beta_star) for x in self.e_vec])
        self.two_ep_minus_bp_vec = numpy.array([2.0 * self.ep_vec[i] - self.bp_vec[i] for i in [0, 1, 2]])
        self.b = math.sqrt(rad_sq(self.b_vec))
        self.e = math.sqrt(rad_sq(self.e_vec))
        self.bp = math.sqrt(rad_sq(self.bp_vec))
        self.ep = math.sqrt(rad_sq(self.ep_vec))

    cpdef complex _integrand(self, double wx, double wy, double wz, double kx, double ky, double kz):
        cdef complex pref = cmath.exp(-1.0j * (kx * self.Ex + ky * self.Ey + kz * self.Ez))
        pref /= (2.0 * math.pi * self.beta_star) ** (3.0 / 2.0)
        cdef double w = (wx ** 2 + wy ** 2 + wz ** 2) ** 0.5
        cdef double factor1 = (2.0 * self.beta_star) / (w * self.b)
        num = (wx - self.ex) ** 2 + (wy - self.ey) ** 2 + (wz - self.ez) ** 2
        num += (w * self.b + wx * self.bx + wy * self.by + wz * self.bz)
        den = 2.0 * self.beta_star
        factor2 = math.exp(- num / den)
        factor3 = cmath.exp(- 1.0j * (wx * kx + wy * ky + wz * kz))
        return pref * factor1 * factor2 * factor3

    def _f_real(self, double wx, double wy, double wz, double kx, double ky, double kz):
        return self._integrand(wx, wy, wz, kx, ky, kz).real

    def _f_imag(self, double wx, double wy, double wz, double kx, double ky, double kz):
        return self._integrand(wx, wy, wz, kx, ky, kz).imag

    def h_corr_quadrature(self, k_vec, L_by_sigma=7.0):
        if L_by_sigma < 2.0:
            print 'WARNING: L < 2 sigma, probably not enough'
        sigma = math.sqrt(self.beta_star)
        r = [- L_by_sigma * sigma, L_by_sigma * sigma]
        options = {'limit': 200, 'points': [0.0], 'epsabs': 1e-10, 'epsrel': 5e-10}
        I_r, err_r = nquad(self._f_real, [r, r, r], args=k_vec, opts=options)
        I_i, err_i = nquad(self._f_imag, [r, r, r], args=k_vec, opts=options)
        return I_r, err_r, I_i, err_i

    def h_corr_analytic(self, k_vec):
        kp_vec = k_vec * math.sqrt(2.0 * self.beta_star)
        V = cmath.sqrt(rad_sq(self.two_ep_minus_bp_vec) -
                       rad_sq(kp_vec) - 2.0j *
                       numpy.dot(self.two_ep_minus_bp_vec, kp_vec))
        I_plus = cmath.exp(-0.5 * self.bp * V) * erfc(0.5 * (self.bp - V))
        I_minus = cmath.exp(0.5 * self.bp * V) * erfc(0.5 * (self.bp + V))
        return (cmath.exp(-1.0j * numpy.dot(k_vec, self.E_vec)) *
                cmath.exp(- self.ep ** 2 + 0.25 * (self.bp ** 2 + V ** 2)) *
                (I_plus - I_minus) / (self.bp * V))

    def full_check(self, k_vec, ID='test'):

        k_vec = numpy.asarray(k_vec)

        hc_a = self.h_corr_analytic(k_vec)
        hc_a_re = hc_a.real
        hc_a_im = hc_a.imag

        with open('output_%s.dat' % ID, 'w') as out:
            out_vect('A', self.A_vec, out)
            out_vect('B', self.B_vec, out)
            out_vect('E', self.E_vec, out)
            out_vect('k', k_vec, out)
            out.write('beta_star: %f\n' % self.beta_star)
            out.write('\n')
            out.flush()
            res = []
            for L_by_sigma in [4.0, 8.0, 12.0]:
                out.write('  L_by_sigma: %f\n' % L_by_sigma)
                t_start = time.clock()
                output = self.h_corr_quadrature(k_vec, L_by_sigma=L_by_sigma)
                hc_q_re, err_hc_q_re, hc_q_im, err_hc_q_im = output[:]
                t_end = time.clock()
                diff_re = abs(hc_a_re - hc_q_re)
                diff_im = abs(hc_a_im - hc_q_im)
                out.write('  Re(hcorr): %+.14f %+.14f [ratio=%.8f, diff=%.2e, err=%.2e] ' % (hc_a_re, hc_q_re, hc_a_re / hc_q_re, diff_re, err_hc_q_re))
                if diff_re < err_hc_q_re:
                    out.write('OK\n')
                else:
                    out.write('FAIL\n')
                out.write('  Im(hcorr): %+.14f %+.14f [ratio=%.8f, diff=%.2e, err=%.2e] ' % (hc_a_im, hc_q_im, hc_a_im / hc_q_im, diff_im, err_hc_q_im))
                if diff_im < err_hc_q_im:
                    out.write('OK\n')
                else:
                    out.write('FAIL\n')
                out.write('  [elapsed: %.1f s]\n' % (t_end - t_start))
                out.write('\n')
                out.flush()
