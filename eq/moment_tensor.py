# import sys
import operator
import functools
from random import random
import unittest
from math import sqrt, cos, sin, acos, atan2
import numpy as np
from numpy import dot

import kshramt


class Error(Exception):
    pass

def _error(cond=True, msg='', *args):
    if cond:
        raise Error(msg.format(*args))


_SQRT2 = sqrt(2)
_INV_SQRT2 = _SQRT2/2


class MomentTensor(object):
    """
    Converter between moment tensor formats.
    Extension is positive.
    Angles are in scale of degree.
    """

    XYZ_SIGN_FROM_RTF = {
        'r': ('z', +1),
        't': ('y', -1),
        'f': ('x', +1),
    }

    _R_yz_from_xx_zz = dot(((0, -1, 0),
                            (1, 0, 0),
                            (0, 0, 1)),
                           ((_INV_SQRT2, 0, _INV_SQRT2),
                            (0, 1, 0),
                            (-_INV_SQRT2, 0, _INV_SQRT2)))
    _R_conjugate_for_yz = dot(((1, 0, 0),
                               (0, 0, 1),
                               (0, -1, 0)),
                              ((-1, 0, 0),
                               (0, -1, 0),
                               (0, 0, 1)))
    def __init__(self):
        self.xx = 0
        self.yy = 0
        self.zz = 0
        self.xy = 0
        self.xz = 0
        self.yz = 0

    def __repr__(self):
        return 'class:{}\t{}'.format(self.__class__.__name__, self.__str__())

    def __str__(self):
        return 'xx:{}\tyy:{}\tzz:{}\txy:{}\txz:{}\tyz:{}'.format(self.xx, self.yy, self.zz, self.xy, self.xz, self.yz)

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.mxyz == other.mxyz

    def __itruediv__(self, x):
        self *= 1/x
        return self

    def __truediv__(self, x):
        ret = type(self)()
        ret.mxyz = self.mxyz
        ret /= x
        return ret

    def __imul__(self, x):
        self.xx *= x
        self.xy *= x
        self.xz *= x
        self.yy *= x
        self.yz *= x
        self.zz *= x
        return self

    def __mul__(self, x):
        ret = type(self)()
        ret.mxyz = self.mxyz
        ret *= x
        return ret

    __rmul__ = __mul__

    def __isub__(self, other):
        self += (-1)*other
        return self


    def __sub__(self, other):
        ret = type(self)()
        ret.mxyz = self.mxyz
        ret -= other
        return ret

    def __iadd__(self, other):
        if isinstance(other, type(self)):
            self.xx += other.xx
            self.xy += other.xy
            self.xz += other.xz
            self.yy += other.yy
            self.yz += other.yz
            self.zz += other.zz
            return self
        else:
            return NotImplemented

    def __add__(self, other):
        if isinstance(other, type(self)):
            ret = type(self)()
            ret.mxyz = self.mxyz
            ret += other
            return ret
        else:
            return NotImplemented

    def magnitude(self, unit='Nm'):
        """
        moment (Nm)
        """
        if unit == 'Nm':
            return (np.log10(self.moment) + 7)/1.5 - 10.7
        elif unit == 'dynecm':
            return np.log10(self.moment)/1.5 - 10.7
        else:
            return NotImplemented

    @property
    def moment(self):
        (m1, _, m3), _ = self.ms_rotateion
        return (m3 - m1)/2

    @property
    def strike_dip_rakes(self):
        return self._strike_dip_rakes(self.mxyz)

    @property
    def strike_dip_rake(self):
        sdr, _ = self.strike_dip_rakes
        return sdr

    @strike_dip_rake.setter
    def strike_dip_rake(self, sdr_m0):
        n_sdr_m0 = len(sdr_m0)
        if n_sdr_m0 == 3:
            strike, dip, rake = sdr_m0
            m0 = 1
        elif n_sdr_m0 == 4:
            strike, dip, rake, m0 = sdr_m0
            _error(m0 < 0, 'm0 < 0: {}', m0)
        else:
            _error(True, 'invalid argument: {}', sdr_m0)

        R = self._dots(self._rotate_xy(-strike),
                       self._rotate_xz(-dip),
                       self._rotate_xy(rake),
                       self._R_yz_from_xx_zz)
        self.mxyz = self._dots(R,
                               ((-m0, 0, 0),
                                (0, 0, 0),
                                (0, 0, m0)),
                               np.transpose(R))

    def _correction_strike_dip_rake(self, strike, dip, rake):
        if dip > 90: # 0 <= dip <= 90
            dip = 180 - dip
            if strike >= 0:
                strike -= 180
            else:
                strike += 180
            rake *= -1
        elif dip == 90: # 0 <= strike < 180 if dip == 90
            if strike < 0:
                strike += 180
                rake *= -1
            if strike == 180:
                strike = 0
                rake *= -1
        if rake == -180: # -180 < rake <= 180
            rake = 180

        return strike, dip, rake

    def _strike_dip_rake_from_R_yz(self, R_yz):
        """
        R_yz is a rotation matrix where

        R_yz*M*transpose(R_yz) = [[0, 0, 0],
                                  [0, 0, 1],
                                  [0, 1, 0]]

        Retured values are (strike, dip, rake), where

        0 <= dip <= 90
        -180 <= strike <= 180
        0 <= strike < 180 if dip == 90
        -180 < rake <= 180
        rake == 0 if dip == 0 (only rake - strike can be resolved)
        """
        R12 = R_yz[0][1]
        R13 = R_yz[0][2]
        R22 = R_yz[1][1]
        R23 = R_yz[1][2]
        R32 = R_yz[2][1]
        R33 = R_yz[2][2] # result is unstable if abs(R33) is small.

        # You can not use elements of 1st column since they have no effect for a resultant momen tensor.
        cos_dip = R33
        if abs(cos_dip) > 1: # todo: is this ok?
            _error(abs(cos_dip) > 1 + 1e-7, 'abs(cos_dip) > 1 + 1e-7: {}', cos_dip)
            cos_dip = _sign(cos_dip)
        dip = np.arccos(cos_dip)
        sin_dip = sin(dip)
        if abs(sin_dip) <= 1e-7: # todo: better threshold
            rake = 0
            strike = np.arctan2(R12, R22)
        else:
            strike = np.arctan2(-R23, R13)
            sin_strike = -R23/sin_dip
            sin_rake = R32/sin_dip
            if _INV_SQRT2 <= abs(sin_strike):
                rake = np.arctan2(sin_rake, (R12 + sin_rake*cos_dip*cos(strike))/sin_strike)
            else:
                rake = np.arctan2(sin_rake, (R22 - sin_rake*sin_strike*cos_dip)/cos(strike))

        return self._correction_strike_dip_rake(np.rad2deg(strike), np.rad2deg(dip), np.rad2deg(rake))

    def _strike_dip_rakes(self, m):
        """
        Returns
        ((strike1, dip1, rake1),
         (strike2, dip2, rake2))
        where (strike2, dip2, rake2) could be unstable.
        """
        _, R = self._sorted_eig(m)
        R_yz = dot(R, np.transpose(self._R_yz_from_xx_zz))
        R_yz_conjugate = dot(R,
                             np.transpose(dot(self._R_conjugate_for_yz,
                                              self._R_yz_from_xx_zz)))
        sdr1 = self._strike_dip_rake_from_R_yz(R_yz)
        sdr2 = self._strike_dip_rake_from_R_yz(R_yz_conjugate)
        if abs(R_yz[2][2]) <= _INV_SQRT2:
            return sdr1, sdr2
        else:
            return sdr2, sdr1

    @property
    def ms_rotateion(self):
        return self._sorted_eig(self.mxyz)

    @classmethod
    def _sorted_eig(cls, m):
        es, vs = np.linalg.eigh(m)
        es, is_ = cls._sorted_i(es)
        return es, np.transpose(cls._update_by_indices(np.transpose(vs), is_))

    @staticmethod
    def _update_by_indices(xs, is_):
        _error(len(xs) != len(is_), 'xs, is_: {}, {}', xs, is_)
        return [xs[i] for i in is_]

    @staticmethod
    def _sorted_i(xs):
        sorted_xs = []
        sorted_indices = []
        for i, x in sorted(enumerate(xs), key=operator.itemgetter(1)):
            sorted_xs.append(x)
            sorted_indices.append(i)
        return sorted_xs, sorted_indices

    # xyz

    yx = property(lambda self: self.xy, lambda self, value: setattr(self, 'xy', value))
    zx = property(lambda self: self.xz, lambda self, value: setattr(self, 'xz', value))
    zy = property(lambda self: self.yz, lambda self, value: setattr(self, 'yz', value))

    @property
    def mxyz(self):
        return ((self.xx, self.xy, self.xz),
                (self.yx, self.yy, self.yz),
                (self.zx, self.zy, self.zz))

    @mxyz.setter
    def mxyz(self, value):
        ((self.xx, self.xy, self.xz),
         (      _, self.yy, self.yz),
         (      _,       _, self.zz)) = value

    @property
    def mxxyyzzxyxzyz(self):
        return self.xx, self.yy, self.zz, self.xy, self.xz, self.yz

    @mxxyyzzxyxzyz.setter
    def mxxyyzzxyxzyz(self, value):
        self.xx, self.yy, self.zz, self.xy, self.xz, self.yz = value

    # rtf

    @property
    def mrtf(self):
        return ((self.rr, self.rt, self.rf),
                (self.tr, self.tt, self.tf),
                (self.fr, self.ft, self.ff))

    @mrtf.setter
    def mrtf(self, value):
        ((self.rr, self.rt, self.rf),
         (      _, self.tt, self.tf),
         (      _,       _, self.ff)) = value

    @property
    def mrrttffrtrftf(self):
        return self.rr, self.tt, self.ff, self.rt, self.rf, self.tf

    @mrrttffrtrftf.setter
    def mrrttffrtrftf(self, value):
        self.rr, self.tt, self.ff, self.rt, self.rf, self.tf = value

    # m1to6

    m1 = property(lambda self: self.xy)
    m2 = property(lambda self: (-self.xx) + self.m6)
    m3 = property(lambda self: -self.xz)
    m4 = property(lambda self: -self.yz)
    m5 = property(lambda self: self.zz - self.m6)
    m6 = property(lambda self: (self.xx + self.yy + self.zz)/3)

    @property
    def m1to6(self):
        return (self.m1, self.m2, self.m3, self.m4, self.m5, self.m6)

    @m1to6.setter
    def m1to6(self, value):
        m1to6 = tuple(value)
        m2 = m1to6[1]
        m5 = m1to6[4]
        m6 = m1to6[5]
        self.xx = m6 - m2
        self.yy = m6 + m2 - m5
        self.zz = m6 + m5
        self.xy = m1to6[0]
        self.xz = -m1to6[2]
        self.yz = -m1to6[3]

    @staticmethod
    def _dots(*ms):
        return functools.reduce(dot, ms)

    @staticmethod
    def _rotate_xy(theta):
        t = np.deg2rad(theta)
        cos_t = cos(t)
        sin_t = sin(t)
        return ((cos_t, -sin_t, 0),
                (sin_t, cos_t, 0),
                (0, 0, 1))

    @staticmethod
    def _rotate_xz(theta):
        t = np.deg2rad(theta)
        cos_t = cos(t)
        sin_t = sin(t)
        return ((cos_t, 0, -sin_t),
                (0, 1, 0),
                (sin_t, 0, cos_t))

    @staticmethod
    def make_rtf_property(rtf1, rtf2):
        xyz1, sign1 = MomentTensor.XYZ_SIGN_FROM_RTF[rtf1]
        xyz2, sign2 = MomentTensor.XYZ_SIGN_FROM_RTF[rtf2]
        xyz = xyz1 + xyz2
        sign_ = sign1*sign2
        return property(lambda self: sign_*getattr(self, xyz),
                        lambda self, value: setattr(self, xyz, sign_*value))

    def amplitude_distribution(self, order=5):
        triangles, points = kshramt.sphere_mesh(n=order, r=1, base=20)
        strike, dip, rake = np.deg2rad(self.strike_dip_rake)
        strike = np.pi/2 - strike
        cos_strike = cos(strike)
        sin_strike = sin(strike)
        Pstrike = [[cos_strike, -sin_strike, 0.0],
                   [sin_strike, cos_strike, 0.0],
                   [0.0, 0.0, 1.0]]
        cos_dip = cos(dip)
        sin_dip = sin(dip)
        Pdip = [[1.0, 0.0, 0.0],
                [0.0, cos_dip, -sin_dip],
                [0.0, sin_dip, cos_dip]]
        cos_rake = cos(rake)
        sin_rake = sin(rake)
        Prake = [[cos_rake, -sin_rake, 0.0],
                 [sin_rake, cos_rake, 0.0],
                 [0.0, 0.0, 1.0]]
        Psdr = self._dots(Pstrike, Pdip, Prake)
        m1, m2, m3 = self.ms_rotateion[0]
        return ([dot(Psdr, xyz) for xyz in points],
                triangles,
                _amplitudes_of_triangles(points, triangles, m2, m3))

_rtf = MomentTensor.XYZ_SIGN_FROM_RTF.keys()
for rtf1 in _rtf:
    for rtf2 in _rtf:
        setattr(MomentTensor, rtf1 + rtf2, MomentTensor.make_rtf_property(rtf1, rtf2))


def merge_amplitude_distribution(points_triangles_amplitudess):
    points = []
    triangles = []
    amplitudes = []
    i = 0
    for p, t, a in points_triangles_amplitudess:
        points.extend(p)
        triangles.extend((i1 + i, i2 + i, i3 + i) for i1, i2, i3 in t)
        i += len(p)
        amplitudes.extend(a)
    return (points, triangles, amplitudes)


def vtk(points, triangles, amplitudes):
    return '\n'.join(['# vtk DataFile Version 3.0',
                      'beachball',
                      'ASCII',
                      'DATASET POLYDATA',
                      'POINTS {} FLOAT'.format(len(points)),
                      '\n'.join('{}\t{}\t{}'.format(x, y, z)
                                for x, y, z in points),
                      'POLYGONS {} {}'.format(len(triangles), 4*len(triangles)),
                      '\n'.join('3\t{}\t{}\t{}'.format(i1, i2, i3)
                                for i1, i2, i3 in triangles),
                      'CELL_DATA {}'.format(len(triangles)),
                      'SCALARS polarity int 1',
                      'LOOKUP_TABLE default',
                      '\n'.join(str(_sign(a))
                                for a in amplitudes)])


def _sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def _amplitudes_of_triangles(points, triangles, m2, m3):
    amplitudes = []
    for i_p1, i_p2, i_p3 in triangles:
        x1, y1, z1 = points[i_p1]
        x2, y2, z2 = points[i_p2]
        x3, y3, z3 = points[i_p3]
        x = (x1 + x2 + x3)/3
        y = (y1 + y2 + y3)/3
        z = (z1 + z2 + z3)/3
        amplitudes.append(_amplitude(x, y, z, m2, m3))
    return amplitudes


def amplitude(x, y, z, m1, m2, m3):
    assert m1 <= m2 <= m3
    m_iso = m1 + m2 + m3
    return _amplitude(x, y, z, m2 - m_iso, m3 - m_iso)


def _amplitude(x, y, z, m2, m3):
    r = sqrt(x*x + y*y + z*z)
    a = x/r
    b = y/r
    c = z/r
    half_c_minus_a = (c - a)/2
    b_inv_sqrt2 = b*_INV_SQRT2
    d = b_inv_sqrt2 - half_c_minus_a
    e = -(a + c)*_INV_SQRT2
    f = half_c_minus_a + b_inv_sqrt2
    # avoiding atan2 or acos here will not improve performance
    return (m3*_amplitude_single(_get_theta(c), _get_phi(a, b)) +
            m2*_amplitude_single(_get_theta(f), _get_phi(d, e)))


def _amplitude_single(theta, phi):
    return sin(2*theta)*cos(phi)


def _get_theta(z):
    return acos(z)


def _get_phi(x, y):
    return atan2(y, x)


class Tester(unittest.TestCase):

    def is_almost_equal_angle(self, x, y):
        delta = 0.01
        if abs(x) > 180 - delta:
            xl = x - delta
            xr = x + delta
            if xl < -180:
                return -180 <= y < xr or 360 + xl < y <= 180
            else:
                return xl < y <= 180 or -180 <= y < xr - 360
        else:
            return abs(x - y) <= delta

    def assert_almost_equal_angle(self, x, y):
        self.assertTrue(self.is_almost_equal_angle(x, y))

    def assert_almost_equal_plane(self, sdr1, sdr2):
        s1, d1, r1 = sdr1
        s2, d2, r2 = sdr2
        s1, d1, r1 = self.m._correction_strike_dip_rake(s1, d1, r1)
        if d1 == 0:
            r1 = 0
            s1 = s1 - r1
        if self.is_almost_equal_angle(d2, 90):
            try:
                self.assert_almost_equal_angle(s1, s2)
                self.assert_almost_equal_angle(d1, d2)
                self.assert_almost_equal_angle(r1, r2)
            except AssertionError:
                s2 = (s2 - 180)%180
                r2 = -r2
                for x, y in ((s1, s2), (d1, d2), (r1, r2)):
                    self.assert_almost_equal_angle(x, y)
        else:
            self.assert_almost_equal_angle(s1, s2)
            self.assert_almost_equal_angle(d1, d2)
            self.assert_almost_equal_angle(r1, r2)

    def assert_one_plane_is_ok(self, sdr, sdrs):
        #print('ORIG:\t{}\t{}\t{}'.format(*sdr), file=sys.stderr)
        def f(sdrs):
            _error(len(sdrs) < 1, 'len(sdrs) < 1: {}', sdrs)
            sdr_ = sdrs[0]
            #print('C:\t{}\t{}\t{}'.format(*sdr_), file=sys.stderr)
            if len(sdrs) == 1:
                self.assert_almost_equal_plane(sdr, sdr_)
            else:
                try:
                    self.assert_almost_equal_plane(sdr, sdr_)
                except AssertionError:
                    f(sdrs[1:])
        f(sdrs)

    def setUp(self):
        self.m = MomentTensor()
        self.m.xx = 1
        self.m.yy = 2
        self.m.zz = 3
        self.m.xy = 4
        self.m.xz = 5
        self.m.yz = 6

    def test_mul_div(self):
        m1 = MomentTensor()
        m2 = MomentTensor()
        m3 = MomentTensor()
        m1.mxxyyzzxyxzyz = (1, 2, 3, 4, 5, 6)
        m2.mxxyyzzxyxzyz = (2, 4, 6, 8, 10, 12)
        m3.mxxyyzzxyxzyz = (3, 6, 9, 12, 15, 18)
        self.assertEqual((m1*2), m2)
        self.assertEqual((2*m1), m2)
        m2 *= 1.5
        self.assertAlmostEqual(m2.m1to6, m3.m1to6)
        self.assertAlmostEqual((m3/3).m1to6, m1.m1to6)
        m3 /= 3
        self.assertAlmostEqual(m3.m1to6, m1.m1to6)

    def test_add_sub(self):
        m1 = MomentTensor()
        m1_ = MomentTensor()
        m2 = MomentTensor()
        m2_ = MomentTensor()
        m3 = MomentTensor()
        m1.mxxyyzzxyxzyz = (2, 4, 6, 8, 10, 12)
        m1_.mxxyyzzxyxzyz = (2, 4, 6, 8, 10, 12)
        m2.mxxyyzzxyxzyz = (1, 3, 5, 7, 9, 11)
        m2_.mxxyyzzxyxzyz = (1, 3, 5, 7, 9, 11)
        m3.mxxyyzzxyxzyz = (3, 7, 11, 15, 19, 23)
        self.assertEqual((m1 + m2), m3)
        self.assertEqual((m3 - m1), m2)
        m2 += m1
        self.assertEqual(m2, m3)
        m3 -= m2_
        self.assertEqual(m3, m1_)

    def test_moment(self):
        self.m.m1to6 = (2, 0, 0, 0, 0, 3)
        self.assertAlmostEqual(self.m.moment, 2)

    def test_set_strike_dip_rake(self):
        for s, d, r, (xx, xy, xz,
                      yx, yy, yz,
                      zx, zy, zz) in ((0, 0, 0, (0, 0, 0,
                                                 0, 0, 1,
                                                 0, 1, 0)),
                                      (90, 0, 0, (0, 0, 1,
                                                  0, 0, 0,
                                                  1, 0, 0)),
                                      (0, 90, 0, (0, 1, 0,
                                                  1, 0, 0,
                                                  0, 0, 0)),
                                      (45, 90, 0, (1, 0, 0,
                                                   0, -1, 0,
                                                   0, 0, 0)),
                                      (0, 45, -90, (1, 0, 0,
                                                    0, 0, 0,
                                                    0, 0, -1)),
                                      (-180, 45, -90, (1, 0, 0,
                                                       0, 0, 0,
                                                       0, 0, -1)),
                                      (180, 45, -90, (1, 0, 0,
                                                      0, 0, 0,
                                                      0, 0, -1))):
            self.m.strike_dip_rake = (s, d, r)
            self.assertAlmostEqual(self.m.xx, xx)
            self.assertAlmostEqual(self.m.xy, xy)
            self.assertAlmostEqual(self.m.xz, xz)
            self.assertAlmostEqual(self.m.yx, yx)
            self.assertAlmostEqual(self.m.yy, yy)
            self.assertAlmostEqual(self.m.yz, yz)
            self.assertAlmostEqual(self.m.zx, zx)
            self.assertAlmostEqual(self.m.zy, zy)
            self.assertAlmostEqual(self.m.zz, zz)

    def test_strike_dip_rake(self):
        for m1to6, (s0, d0, r0) in (((1, 0, 0, 0, 0, 0), (0, 90, 0)),
                                    ((0, 2, 0, 0, 0, 0), (45, 90, 180)),
                                    ((0, 0, 3, 0, 0, 0), (0, 90, -90)),
                                    ((0, 0, 0, 4, 0, 0), (90, 90, 90)),
                                    ((0, 0, 0, 0, 5, 0), (90, 45, 90)),
                                    ((0, 0, 0, 0, 0, 6), (-180, 45, 90))):
            self.m.m1to6 = m1to6
            s, d, r = self.m.strike_dip_rake
            self.assert_almost_equal_plane((s, d, r), (s0, d0, r0))

        for s, d, r in ((45, 90, 0),
                        (45, 90, 1),
                        (45, 90, 180),
                        (0, 90, 180)):
            self.m.strike_dip_rake = (s, d, r)
            s_, d_, r_ = self.m.strike_dip_rake
            self.assert_almost_equal_plane((s_, d_, r_), (s, d, r))

        for _ in range(500):
            s = 360*(0.5 - random())
            self.m.strike_dip_rake = (s, 0, 0)
            self.assert_one_plane_is_ok((s, 0, 0), self.m.strike_dip_rakes)
            self.m.strike_dip_rake = (s, 90, 0)
            self.assert_one_plane_is_ok((s, 90, 0), self.m.strike_dip_rakes)

        for sdr in ((180, 90, 1),):
            self.m.strike_dip_rake = sdr
            sdr1, sdr2 = self.m.strike_dip_rakes
            s_, d_, r_ = self.m._correction_strike_dip_rake(*sdr)
            if d_ == 0:
                r_ = 0
                s_ = s_ - r_
            self.assert_one_plane_is_ok((s_, d_, r_), (sdr1, sdr2))

        for _ in range(500):
            s = 360*(random() - 0.5)
            d = min(90.001*random(), 90)
            r = 360*(random() - 0.5)
            s_, d_, r_ = self.m._correction_strike_dip_rake(s, d, r)
            if d_ == 0:
                r_ = 0
                s_ = s_ - r_
            self.m.strike_dip_rake = (s, d, r)
            sdr1, sdr2 = self.m.strike_dip_rakes

    def test__sorted_eig(self):
        es, vs = self.m._sorted_eig(((2, 0, 0),
                                     (0, 0, 0),
                                     (0, 0, 1)))
        self.assertAlmostEqual(es, [0, 1, 2])
        ans = ((0, 0, 1),
               (1, 0, 0),
               (0, 1, 0))
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(vs[i][j], ans[i][j])

    def test__update_by_indices(self):
        xs = self.m._update_by_indices([0, 1, 2], [0, 2, 1])
        self.assertEqual(xs, [0, 2, 1])

    def test__sorted_i(self):
        xs, is_ = self.m._sorted_i([10, 30, 20])
        self.assertEqual(xs, [10, 20, 30])
        self.assertEqual(is_, [0, 2, 1])

    def test_mrtf(self):
        orig = self.m.mrtf
        self.m.mrtf = orig
        self.assertAlmostEqual(self.m.mrtf, orig)

    def test_mxyz(self):
        orig = self.m.mxyz
        self.m.mxyz = orig
        self.assertAlmostEqual(self.m.mxyz, orig)

    def test_m1to6(self):
        m1to6 = (1, 2, 3, 4, 5, 6)
        self.m.m1to6 = m1to6
        self.assertAlmostEqual(self.m.m1to6, m1to6)

    def test_xyz(self):
        self.assertEqual(self.m.xx, 1)
        self.assertEqual(self.m.yy, 2)
        self.assertEqual(self.m.zz, 3)
        self.assertEqual(self.m.xy, 4)
        self.assertEqual(self.m.xz, 5)
        self.assertEqual(self.m.yz, 6)

    def test_rtf(self):
        self.assertEqual(self.m.rr, 3)
        self.assertEqual(self.m.tt, 2)
        self.assertEqual(self.m.ff, 1)
        self.assertEqual(self.m.rt, -6)
        self.assertEqual(self.m.rf, 5)
        self.assertEqual(self.m.tf, -4)

    def test_xyz_rtf(self):
        self.assertEqual(self.m.rr, self.m.zz)
        self.assertEqual(self.m.tt, self.m.yy)
        self.assertEqual(self.m.ff, self.m.xx)
        self.assertEqual(self.m.tr, -self.m.yz)
        self.assertEqual(self.m.fr, self.m.xz)
        self.assertEqual(self.m.ft, -self.m.xy)

    def test_symmetry(self):
        self.assertEqual(self.m.xy, self.m.yx)
        self.assertEqual(self.m.xz, self.m.zx)
        self.assertEqual(self.m.yz, self.m.zy)

        self.assertEqual(self.m.rt, self.m.tr)
        self.assertEqual(self.m.rf, self.m.fr)
        self.assertEqual(self.m.tf, self.m.ft)

        self.m.yx = 10
        self.assertEqual(self.m.xy, self.m.yx)
        self.m.zx = 20
        self.assertEqual(self.m.xy, self.m.yx)
        self.m.zy = 30
        self.assertEqual(self.m.xy, self.m.yx)

        self.m.tr = 40
        self.assertEqual(self.m.rt, self.m.tr)
        self.m.fr = 50
        self.assertEqual(self.m.rt, self.m.tr)
        self.m.ft = 60
        self.assertEqual(self.m.rt, self.m.tr)

    def test_1to6(self):
        self.assertEqual(self.m.m1, 4)
        self.assertEqual(self.m.m2, 1)
        self.assertEqual(self.m.m3, -5)
        self.assertEqual(self.m.m4, -6)
        self.assertEqual(self.m.m5, 1)
        self.assertEqual(self.m.m6, 2)


if __name__ == '__main__':
    unittest.main()
