import unittest
import math

import eq.kshramt


def _parse_record_type(s):
    assert s in 'JUI'
    return s


def _parse_latitude(s):
    d = int(s[:3])
    m = int(s[3:])/100
    return d + math.copysign(m/60, d)


def _parse_longitude(s):
    d = int(s[:4])
    m = int(s[4:])/100
    return d + math.copysign(m/60, d)


def _parse_depth(s):
    if s.endswith('  '):
        return int(s)
    else:
        return int(s)/100


_ordA = ord('A')
def _parse_magnitude(s):
    if s[0] in '-0123456789':
        return int(s)/10
    else:
        return (_ordA - ord(s[0]) - 1) - int(s[1])/10


def load(fp):
    return list(load_lazy(fp))


def load_lazy(fp):
    return map(parse_record, fp)


def parse_record(line):
    """
    see:
    - http://data.sokki.jmbsc.or.jp/cdrom/seismological/data/format/hypfmt_e.htm
    - ftp://ftp.eri.u-tokyo.ac.jp/pub/data/jma/mirror/JMA_HYP/format_e.txt
    """

    assert line[0] in 'JUIC'
    if line[0] == 'C':
        return {'record_type': line[0], 'comment': line[1:]}
    else:
        return _parse_record(line)


_parse_record = eq.kshramt.make_parse_fixed_width((
    # J: JMA
    # U: USGS
    # I: ISC/IASPEI/etc.
    ('record_type', 1, _parse_record_type),
    ('year', 4, int),
    ('month', 2, int),
    ('day', 2, int),
    ('hour', 2, int),
    ('minute', 2, int),
    ('second', 4, lambda s: int(s)/100),
    ('second_standard_error', 4, lambda s: int(s)/100),
    ('latitude', 7, _parse_latitude), # degree
    ('latitude_standard_error', 4, lambda s: int(s)/100/60), # degree
    ('longitude', 8, _parse_longitude), # degree
    ('longitude_standard_error', 4, lambda s: int(s)/100/60), # degree
    ('depth', 5, _parse_depth), # km
    ('depth_error', 3, lambda s: int(s)/100), # km
    # first JMA magunitude or body wave magnitude by USGS
    ('magnitude_1', 2, _parse_magnitude), 
    # 'J': Tsuboi's displacement magnitude (Mj)
    # 'D': displacement magnitude
    # 'd': same as 'D', but the number of the used stations is less than 3
    # 'V': velocity magnitude
    # 'v': same as 'V', but the number of the used stations is less than 4
    # 'B': USGS body wave magnitude
    # 'S': USGS surface wave magnitude
    ('magnitude_1_type', 1, str),
    # second JMA magunitude or surface wave magnitude by USGS
    ('magnitude_2', 2, _parse_magnitude),
    ('magnitude_2_type', 1, str),
    # ' ': determined by another agency
    # '1': standard table such as 83A
    # '2': table of far east off the Sanriku district
    # '3': table of the east off Hokkaido district
    # '4': table of the regions of southern parts of the Kurile Islands (with 83A travel time table)
    # '5': standard table (JMA 2001)
    # '6': table of the regions of southern of parts of the Kurile Islands (with JMA2001 travel time table)
    ('travel_time_table', 1, str),
    # 1: depth-free method
    # 2: depth-slice method
    # 3: fixed depth
    # 4: using depth phases
    # 5: using S-P time
    # 6: poor solution
    # 7: not determined or not accepted
    ('precision_of_hypocenter', 1, int),
    # 1: natural earthquake
    # 2: insufficient number of JMA stations
    # 3: artificial event
    # 4: noise
    # 5: low frequency earthquake
    ('subsidiary_information', 1, int),
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
    ('maximum_intensity', 1, str),
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
    ('damage_class', 1, str),
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
    ('tsunami_class', 1, str),
    ('district number', 1, int), # district number of epicenter
    ('region_number', 3, int), # geographical region number of epicenter
    ('region name', 24, str), # geographical region name of epicenter
    ('number_of_stations', 3, int), # number of stations contributed to the hypocenter determination
    # K: high-precision hypocenters
    # S: low-precision hypocenters
    ('hypocenter_determination_flag', 1, str),
))


class Tester(unittest.TestCase):

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


if __name__ == '__main__':
    unittest.main()
