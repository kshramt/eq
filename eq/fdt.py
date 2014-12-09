"""
[FDT format](http://web.archive.org/web/20141002074925/http://www.kueps.kyoto-u.ac.jp/~web-bs/tsg/software/FDTformat/)
"""


import unittest
import math

import eq.kshramt


HALF_PI = math.pi/2
TWO_PI = math.pi*2
FAULTING_TYPES = ('n', 'r', 'd', 's')


def to_strike_dip_rake(f_az, f_pl, s_az, s_pl, faulting_type, comment=None):
    assert faulting_type in FAULTING_TYPES
    strike = f_az - HALF_PI
    if strike > math.pi:
        strike -= TWO_PI
    dip = f_pl
    s_az = HALF_PI - s_az
    cos_s = math.cos(s_pl)
    s_x = cos_s*math.cos(s_az)
    s_y = cos_s*math.sin(s_az)
    f_x = math.sin(strike)
    f_y = math.cos(strike)
    rake = -math.acos(s_x*f_x + s_y*f_y)
    if faulting_type == 'n':
        pass
    elif faulting_type == 'r':
        rake += math.pi
    elif faulting_type == 'd':
        if rake > -HALF_PI:
            rake += math.pi
    elif faulting_type == 's':
        if rake < -HALF_PI:
            rake += math.pi
    else:
        error('must not happen')
    return strike, dip, rake


def load(fp, fail_fn=None):
    if fail_fn is None:
        for line in fp:
            yield parse_record(line)
    else:
        for line in fp:
            try:
                yield parse_record(line)
            except Exception as e:
                is_yield, val = fail_fn(line, e)
                if is_yield:
                    yield val


def parse_record(line):
    """
    input: degree
    output: radian
    """
    f_az, f_pl, s_az, s_pl, faulting_type, *comment = line.rstrip('\n').split(maxsplit=5)
    faulting_type = faulting_type.lower()
    assert faulting_type in FAULTING_TYPES
    ret = dict(f_az=eq.kshramt.rad(float(f_az)),
               f_pl=eq.kshramt.rad(float(f_pl)),
               s_az=eq.kshramt.rad(float(s_az)),
               s_pl=eq.kshramt.rad(float(s_pl)),
               faulting_type=faulting_type)
    if comment:
        ret['comment'] = comment[0]
    return ret


class Error(Exception):
    pass


def error(msg=''):
    raise Error(msg)


class _Tester(unittest.TestCase):

    def test_parse_record(self):
        self.assertAlmostEqual(parse_record('10.0 -20 30 40 N'),
                               dict(f_az=eq.kshramt.rad(10),
                                    f_pl=eq.kshramt.rad(-20),
                                    s_az=eq.kshramt.rad(30),
                                    s_pl=eq.kshramt.rad(40),
                                    faulting_type='n'))
        self.assertAlmostEqual(parse_record('10.0 -20 30 40 N a comment'),
                               dict(f_az=eq.kshramt.rad(10),
                                    f_pl=eq.kshramt.rad(-20),
                                    s_az=eq.kshramt.rad(30),
                                    s_pl=eq.kshramt.rad(40),
                                    faulting_type='n',
                                    comment='a comment'))


if __name__ == '__main__':
    unittest.main()
