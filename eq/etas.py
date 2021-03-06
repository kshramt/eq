#!/usr/bin/python


import unittest
import sys
import random
from math import exp, log

import numpy as np


def gen_etas_seq(t_end, mu, k, c, alpha, p, mr, tau, mag_fn):
    """ETAS event sequence"""
    assert t_end > 0
    assert mu > 0
    assert k >= 0
    assert c > 0
    assert tau > 0
    mu_tau = mu/tau
    k_tau = k/_omori_integrate(tau, c, p)
    parent = 0
    current = parent
    queue = []
    for _ in range(_n_events(mu_tau*t_end)):
        t = t_end*random.random()
        m = mag_fn()
        current += 1
        pair = (t, m, current, parent)
        yield pair
        queue.append(pair)
    for t_parent, m_parent, parent, _ in queue:
        tail_size = _omori_integrate(t_end - t_parent, c, p)
        coeff = k_tau*exp(alpha*(m_parent - mr))
        n_expect = coeff*tail_size
        for _ in range(_n_events(n_expect)):
            integ = tail_size*(1 - random.random()) # should not be 0
            t = _omori_time(integ, c, p) + t_parent
            assert t <= t_end
            m = mag_fn(m_parent)
            current += 1
            pair = (t, m, current, parent)
            yield pair
            queue.append(pair)


_log10 = log(10)

def random_gr(b, m_min, m_max=sys.float_info.max):
    """random sampling from the Gutenberg-Richter law
    return: [m_min, m_max)
    """
    assert m_min < m_max
    b_log10 = b*_log10
    return -1/b_log10*log(exp(-b_log10*m_min) -
                          (1 - random.random())*
                          (exp(-b_log10*m_min) -
                           exp(-b_log10*m_max)))


def _n_events(n):
    return np.random.poisson(n)


def _omori_time(integ, c, p):
    """_omori_integrate(ret, c, p) == integ
    """
    if p == 1:
        return c*(exp(integ) - 1)
    else:
        return (integ*(1 - p) + c**(1 - p))**(1/(1 - p)) - c


def _omori_integrate(t, c, p):
    """$\int_{0}^{t} 1/(t + c)^{p}$
    """
    if p == 1:
        return log((t + c)/c)
    else:
        return ((t + c)**(1 - p) - c**(1 - p))/(1 - p)


class _Tester(unittest.TestCase):

    def test__omori_time(self):
        t = 2
        c = 3
        p = 1
        integ = _omori_integrate(t, c, p)
        self.assertAlmostEqual(_omori_time(integ, c, p), t)
        t = 2
        c = 3
        p = 4
        integ = _omori_integrate(t, c, p)
        self.assertAlmostEqual(_omori_time(integ, c, p), t)


if __name__ == '__main__':
    unittest.main()
