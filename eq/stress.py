"""
Assume tension is positive.
"""


import unittest
from math import sin, cos, acos

import numpy as np
import numpy.testing as npt

import eq.util
import eq.kshramt


def rake_optimum(P, strike, dip):
    n = normal_from_strike_dip(strike, dip)
    t = traction(P, n)
    v = t - np.dot(t, n)*n
    return eq.kshramt.sign(v[2])*acos((sin(strike)*v[0] + cos(strike)*v[1])/np.linalg.norm(v))


def traction(P, n):
    return np.dot(P, n)


def normal_from_strike_dip(strike, dip):
    sd = sin(dip)
    return np.array((cos(strike)*sd, -sin(strike)*sd, cos(dip)))


def from_alpha_beta_gamma_phi(alpha, beta, gamma, phi):
    assert 0 <= phi <= 1
    inv_2_m_phi = 1e0/(2e0 - phi)
    D = ((1e0, 0e0, 0e0),
         (0e0, (2e0*phi - 1e0)*inv_2_m_phi, 0e0),
         (0e0, 0e0, -(1e0 + phi)*inv_2_m_phi))
    R = _R_alpha_beta_gamma(alpha, beta, gamma)
    return eq.util.dots(R, D, R.T)


def _R_alpha_beta_gamma(alpha, beta, gamma):
    Ca = cos(alpha)
    Sa = sin(alpha)
    Cb = cos(beta)
    Sb = sin(beta)
    Cg = cos(gamma)
    Sg = sin(gamma)
    return np.array(((Ca*Cb, -(Sa*Cg + Sb*Sg*Ca), Sa*Sg - Sb*Ca*Cg),
                     (Sa*Cb, Ca*Cg - Sa*Sb*Sg, -(Sa*Sb*Cg + Sg*Ca)),
                     (Sb, Sg*Cb, Cb*Cg)))


def _R_alpha(x):
    c = cos(x)
    s = sin(x)
    return np.array(((c, -s, 0e0),
                     (s, c, 0e0),
                     (0e0, 0e0, 1e0)))


def _R_beta(x):
    c = cos(x)
    s = sin(x)
    return np.array(((c, 0e0, -s),
                     (0e0, 1e0, 0e0),
                     (s, 0e0, c)))


def _R_gamma(x):
    c = cos(x)
    s = sin(x)
    return np.array(((1e0, 0e0, 0e0),
                     (0e0, c, -s),
                     (0e0, s, c)))


def _test():
    alpha = 0.1
    beta = 0.2
    gamma = 0.3
    npt.assert_almost_equal(_R_alpha_beta_gamma(alpha, beta, gamma),
                            eq.util.dots(_R_alpha(alpha), _R_beta(beta), _R_gamma(gamma)))


class _Tester(unittest.TestCase):

    def test_rake_optimum(self):
        for dip in np.linspace(0.0001, np.pi/2 - 0.0001):
            stress = from_alpha_beta_gamma_phi(0, np.pi, np.pi/2, 0.5)
            self.assertAlmostEqual(rake_optimum(stress, np.pi/2, dip), np.pi/2)


if __name__ == '__main__':
    _test()
    unittest.main()
