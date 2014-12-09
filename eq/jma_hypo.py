import unittest
import datetime

import eq.kshramt


def none(f):
    def none_f(s):
        if str.strip(s):
            return f(s)
    return none_f


def zero(s):
    _s = s.lstrip()
    if _s:
        return _s.replace(' ', '0')
    else:
        return '0'


def _parse_record_type(s):
    assert s in 'JUI', s
    return s


def _parse_time(s):
    y = int(zero(s[:4]))
    m = int(zero(s[4:6]))
    d = int(zero(s[6:8]))
    H = int(zero(s[8:10]))
    M = int(zero(s[10:12]))
    S = int(zero(s[12:14]))
    μS = int(zero(s[14:16]))*10000
    return datetime.datetime(y, m, d, H, M, S, μS)


def _parse_latitude(s):
    sign = 1
    if s.startswith('-'):
        sign = -1
    d = int(zero(s[1:3]))
    m = int(zero(s[3:]))/100
    ret = sign*(d + m/60)
    assert -90 <= ret <= 90, s
    return ret


def _parse_longitude(s):
    sign = 1
    if s.startswith('-'):
        sign = -1
    d = int(zero(s[1:4]))
    m = int(zero(s[4:]))/100
    ret = sign*(d + m/60)
    assert -180 <= ret <= 180, s
    return ret


def _parse_depth(s):
    if s.endswith('  '):
        ret = int(zero(s[:3]))
    else:
        ret = int(zero(s))/100
    assert 0 <= ret <= 1000, s
    return ret

_ordA = ord('A')
@none
def _parse_magnitude(s):
    if s[0] in ' -0123456789':
        ret = int(s)/10
    else:
        ret = (_ordA - ord(s[0]) - 1) - int(s[1])/10
    assert ret <= 16, s
    return ret


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
    see:
    - http://data.sokki.jmbsc.or.jp/cdrom/seismological/data/format/hypfmt_e.htm
    - ftp://ftp.eri.u-tokyo.ac.jp/pub/data/jma/mirror/JMA_HYP/format_e.txt
    """

    if line[0] == 'C':
        return {'record_type': line[0], 'comment': line[1:]}
    else:
        return _parse_record(line)


_parse_record = eq.kshramt.make_parse_fixed_width((
    # J: JMA
    # U: USGS
    # I: ISC/IASPEI/etc.
    ('record_type', 1, _parse_record_type),
    ('time', 16, _parse_time),
    ('second_standard_error', 4, none(lambda s: int(zero(s))/100)),
    ('latitude', 7, _parse_latitude), # degree
    ('latitude_standard_error', 4, none(lambda s: int(zero(s))/100/60)), # degree
    ('longitude', 8, _parse_longitude), # degree
    ('longitude_standard_error', 4, none(lambda s: int(zero(s))/100/60)), # degree
    ('depth', 5, _parse_depth), # km
    ('depth_error', 3, none(lambda s: int(zero(s))/100)), # km
    # first JMA magunitude or body wave magnitude by USGS
    ('magnitude_1', 2, _parse_magnitude), 
    # 'J': Tsuboi's displacement magnitude (Mj)
    # 'D': displacement magnitude
    # 'd': same as 'D', but the number of the used stations is less than 3
    # 'V': velocity magnitude
    # 'v': same as 'V', but the number of the used stations is less than 4
    # 'B': USGS body wave magnitude
    # 'S': USGS surface wave magnitude
    ('magnitude_1_type', 1, none(str)),
    # second JMA magunitude or surface wave magnitude by USGS
    ('magnitude_2', 2, _parse_magnitude),
    ('magnitude_2_type', 1, none(str)),
    # ' ': determined by another agency
    # '1': standard table such as 83A
    # '2': table of far east off the Sanriku district
    # '3': table of the east off Hokkaido district
    # '4': table of the regions of southern parts of the Kurile Islands (with 83A travel time table)
    # '5': standard table (JMA 2001)
    # '6': table of the regions of southern of parts of the Kurile Islands (with JMA2001 travel time table)
    ('travel_time_table', 1, none(str)),
    # 1: depth-free method
    # 2: depth-slice method
    # 3: fixed depth
    # 4: using depth phases
    # 5: using S-P time
    # 6: poor solution
    # 7: not determined or not accepted
    ('precision_of_hypocenter', 1, none(str)),
    # 1: natural earthquake
    # 2: insufficient number of JMA stations
    # 3: artificial event
    # 4: noise
    # 5: low frequency earthquake
    ('subsidiary_information', 1, none(str)),
    # 1-7: 1-7
    # A: 5 lower
    # B: 5 upper
    # C: 6 lower
    # D: 6 upper
    # R: remarkable earthquake; distance of the furthest point where shock was felt is greater than 300 km
    # M: moderate earthquake; 200 km < felt distance < 300 km
    # S: earthquake of small felt area: 100 km < felt distance < 200 km
    # L: local earthquake; felt distance < 100 km
    # F: felt earthquake
    # X: shock is felt by some people but not by JMA observers
    ('maximum_intensity', 1, none(str)),
    # after Utsu
    # 1 : slight damage (cracks on walls and ground)
    # 2 : light damaged (broken houses, roads, etc.)
    # 3 : 2 - 19 persons killed or 2--999 houses completely destroyed
    # 4 : 20--199 persons killed or 1,000--9,999 houses completely destroyed
    # 5 : 200--1,999 persons killed or 10,000--99,999 houses completely destroyed
    # 6 : 2,000--19,999 persons killed or 100,000--999,999 houses completely destroyed
    # 7 : 20,000 or more persons killed or 1,000,000 or more houses completely destroyed
    # X : Injuries or damage were caused but the grade was not clear
    # Y : Injuries and damage are included in the grade for the preceding or following event
    ('damage_class', 1, none(str)),
    # 1929--1988 Utsu's Tsunami class
    # 1: tsunami was observed by tide gage, but it had no damage
    # T: Tsunami was generated
    # 1989-- Imamura and Iida's Tsunami class (height / damage)
    # 1: 50 cm / none
    # 2: 1 m / very slight damage
    # 3: 2 m / slight damage to coastal areas and ship
    # 4: 4--6 m / human injuries
    # 5: 10--20 m / damage to more than 400 km of coastline
    # 6: 30 m / damage to more than 500 km of coastline
    ('tsunami_class', 1, none(str)),
    ('district_number', 1, none(int)), # district number of epicenter
    ('region_number', 3, none(int)), # geographical region number of epicenter
    ('region_name', 24, none(str.strip)), # geographical region name of epicenter
    ('number_of_stations', 3, none(int)), # number of stations contributed to the hypocenter determination
    # K: high-precision hypocenters
    # S: low-precision hypocenters
    ('hypocenter_determination_flag', 1, none(str)),
))


class Tester(unittest.TestCase):

    def test_parse_record(self):
        parse_record('U199711081924569     -220678     1793462    606      2B         9   SOUTH OF FIJI               ')
        parse_record('J192408012201         35         13930        0     65J    325Y     SAGAMI BAY ?              5K')
        parse_record('J1998110103173692 030 271908 116 1294640 166 99     16v   521   7296NEAR AMAMI-OSHIMA ISLAND  4K')
        parse_record('U199811012323016     - 90054     1502884     33     44B         9   E NEW GUINEA REG.,P.N.G.    ')

    def test__parse_time(self):
        self.assertAlmostEqual(_parse_time('2014012013243258'), datetime.datetime(2014, 1, 20, 13, 24, 32, 580000))

    def test_parse_latitude(self):
        self.assertAlmostEqual(_parse_latitude(' 123456'), 12.576)
        self.assertAlmostEqual(_parse_latitude(' 023456'), 2.576)
        self.assertAlmostEqual(_parse_latitude('-123456'), -12.576)
        self.assertAlmostEqual(_parse_latitude('-023456'), -2.576)

    def test_parse_longitude(self):
        self.assertAlmostEqual(_parse_longitude(' 1123456'), 112.576)
        self.assertAlmostEqual(_parse_longitude(' 0123456'), 12.576)
        self.assertAlmostEqual(_parse_longitude('-1123456'), -112.576)
        self.assertAlmostEqual(_parse_longitude('-0123456'), -12.576)

    def test_parse_depth(self):
        self.assertAlmostEqual(_parse_depth('123  '), 123)
        self.assertAlmostEqual(_parse_depth(' 1234'), 1234e-2)
        self.assertAlmostEqual(_parse_depth('01234'), 1234e-2)

    def test__parse_magnitude(self):
        self.assertAlmostEqual(_parse_magnitude('32'), 32e-1)
        self.assertAlmostEqual(_parse_magnitude('02'), 2e-1)
        self.assertAlmostEqual(_parse_magnitude('-1'), -1e-1)
        self.assertAlmostEqual(_parse_magnitude('-9'), -9e-1)
        self.assertAlmostEqual(_parse_magnitude('A0'), -1)
        self.assertAlmostEqual(_parse_magnitude('A9'), -19e-1)
        self.assertAlmostEqual(_parse_magnitude('B0'), -2)
        self.assertAlmostEqual(_parse_magnitude('C0'), -3)
        self.assertAlmostEqual(_parse_magnitude('C1'), -31e-1)
        self.assertAlmostEqual(_parse_magnitude('  '), None)
        self.assertAlmostEqual(_parse_magnitude(' 2'), 0.2)


if __name__ == '__main__':
    unittest.main()
