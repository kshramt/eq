#!/usr/bin/python


import unittest
# import sys
from math import sin, cos, acos
# import re
# import pickle

import numpy as np
# import scipy as sp
# import matplotlib.pyplot as plt
# import pandas as pd

import eq.util
import eq.kshramt


def rake_optimum(P, strike, dip):
    n = normal_from_strike_dip(strike, dip)
    t = traction(P, n)
    v = t - np.dot(t, n)*n
    return eq.kshramt.sign(v[2])*acos(np.dot(v, (sin(strike), cos(strike), 0e0))/np.linalg.norm(v))


def traction(P, n):
    return np.dot(P, n)


def normal_from_strike_dip(strike, dip):
    R_strike = np.array(((cos(strike), sin(strike), 0e0),
                         (-sin(strike), cos(strike), 0e0),
                         (0e0, 0e0, 1e0)))
    R_dip = np.array(((cos(dip), 0e0, sin(dip)),
                      (0e0, 1e0, 0e0),
                      (-sin(dip), 0e0, cos(dip))))
    return eq.util.dots(R_strike, R_dip, (0e0, 0e0, 1e0))


def from_alpha_beta_gamma_phi(alpha, beta, gamma, phi):
    assert 0 <= phi <= 1
    D = np.array(((1e0, 0e0, 0e0),
                  (0e0, (2e0*phi - 1e0)/(2e0 - phi), 0e0),
                  (0e0, 0e0, -(1e0 + phi)/(2e0 - phi))))
    R_alpha = np.array(((cos(alpha), -sin(alpha), 0e0),
                        (sin(alpha), cos(alpha), 0e0),
                        (0e0, 0e0, 1e0)))
    R_beta = np.array(((cos(beta), 0e0, -sin(beta)),
                       (0e0, 1e0, 0e0),
                       (sin(beta), 0e0, cos(beta))))
    R_gamma = np.array(((1e0, 0e0, 0e0),
                        (0e0, cos(gamma), -sin(gamma)),
                        (0e0, sin(gamma), cos(gamma))))
    R = eq.util.dots(R_alpha, R_beta, R_gamma)
    return eq.util.dots(R, D, R.T)


class _Tester(unittest.TestCase):

    def test_(self):
        pass


if __name__ == '__main__':
    unittest.main()
