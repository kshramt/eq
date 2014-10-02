"""
[FDT format](http://web.archive.org/web/20141002074925/http://www.kueps.kyoto-u.ac.jp/~web-bs/tsg/software/FDTformat/)
"""


import unittest

import eq.kshramt
def load(fp):
    return (parse_record(line.rstrip('\n')) for line in fp)


def parse_record(line):
    f_az, f_pl, s_az, s_pl, faulting_type, *comment = line.split(maxsplit=5)
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
