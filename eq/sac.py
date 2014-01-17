"""
# REFERENCES
- [SAC User Manual/SAC Data Format](http://www.iris.edu/software/sac/manual/file_format.html)
"""

import struct

import pylab

FLOAT = pylab.float32
INTEGER = pylab.int32
COMPLEX = pylab.complex64

INTEGER_MIN = pylab.iinfo(INTEGER).min
INTEGER_MAX = pylab.iinfo(INTEGER).max
FLOAT_MIN = pylab.finfo(FLOAT).min
FLOAT_MAX = pylab.finfo(FLOAT).max


def _convert_from_1(b, a):
    def wrapper(f):
        def new_f(x):
            if x == a:
                return b
            else:
                return f(x)
        return new_f
    return wrapper
_none_from_undefined_1 = lambda undefined: _convert_from_1(None, undefined)
_undefined_from_none_1 = lambda undefined: _convert_from_1(undefined, None)


def _convert_from_2(b, a):
    def wrapper(f):
        def new_f(x, y):
            if y == a:
                return b
            else:
                return f(x, y)
        return new_f
    return wrapper
_none_from_undefined_2 = lambda undefined: _convert_from_2(None, undefined)
_undefined_from_none_2 = lambda undefined: _convert_from_2(undefined, None)


def _assert_range_1(lower, upper):
    def wrapper(f):
        def new_f(x):
            assert lower <= x <= upper
            return f(x)
        return new_f
    return wrapper


def _assert_range_2(lower, upper):
    def wrapper(f):
        def new_f(x, y):
            assert lower <= y <= upper
            return f(x, y)
        return new_f
    return wrapper


class FieldProp(object):

    INTERNAL_BYTES_FROM_TYPE = {
        'short_string': 8,
        'long_string': 16,
    }
    BYTES_SHORT_STRING = INTERNAL_BYTES_FROM_TYPE['short_string']
    BYTES_LONG_STRING = INTERNAL_BYTES_FROM_TYPE['long_string']

    ASCII_BYTES_FROM_TYPE = {
        'logical': 10,
        'integer': 10,
        'float': 15,
        'enum': 10,
        'short_string': BYTES_SHORT_STRING,
        'long_string': BYTES_LONG_STRING,
    }

    ASCII_FORMAT_FROM_TYPE = {
        'logical': '{{:>{}d}}'.format(ASCII_BYTES_FROM_TYPE['logical']),
        'integer': '{{:>{}d}}'.format(ASCII_BYTES_FROM_TYPE['integer']),
        'float': '{{:>#{}.7g}}'.format(ASCII_BYTES_FROM_TYPE['float']),
        'enum': '{{:>{}d}}'.format(ASCII_BYTES_FROM_TYPE['enum']),
        'short_string': '{:s}', # assume internal values to always have BYTES_SHORT_STRING bytes
        'long_string': '{:s}', # assume internal values to always have BYTES_LONG_STRING bytes
    }
    PARSE_ASCII_FROM_TYPE = {
        'logical': int,
        'integer': int,
        'float': float,
        'enum': int,
        'short_string': lambda s: s,
        'long_string': lambda s: s,
    }
    BINARY_FORMAT_FROM_TYPE = {
        'logical': 'i',
        'integer': 'i',
        'float': 'f',
        'enum': 'i',
        'short_string': '{}c'.format(BYTES_SHORT_STRING),
        'long_string': '{}c'.format(BYTES_LONG_STRING),
    }

    UNDEFINED_FROM_TYPE = {
        'logical': -12345,
        'integer': -12345,
        'float': -12345.0,
        'enum': -12345,
        'short_string': '-12345  ',
        'long_string': '-12345          ',
    }
    UNDEFINED_LOGICAL = UNDEFINED_FROM_TYPE['logical']
    UNDEFINED_INTEGER = UNDEFINED_FROM_TYPE['integer']
    UNDEFINED_FLOAT = UNDEFINED_FROM_TYPE['float']
    UNDEFINED_ENUM = UNDEFINED_FROM_TYPE['enum']
    UNDEFINED_SHORT_STRING = UNDEFINED_FROM_TYPE['short_string']
    UNDEFINED_LONG_STRING = UNDEFINED_FROM_TYPE['long_string']

    ENUMS = (
        'itime', 'irlim', 'iamph', 'ixy', 'iunkn',
        'idisp', 'ivel', 'iacc', 'ib', 'iday',
        'io', 'ia', 'it0', 'it1', 'it2',
        'it3', 'it4', 'it5', 'it6', 'it7',
        'it8', 'it9', 'iradnv', 'itannv', 'iradev',
        'itanev', 'inorth', 'ieast', 'ihorza', 'idown',
        'iup', 'illlbb', 'iwwsn1', 'iwwsn2', 'ihglp',
        'isro', 'inucl', 'ipren', 'ipostn', 'iquake',
        'ipreq', 'ipostq', 'ichem', 'iother', 'igood',
        'iglch', 'idrop', 'ilowsn', 'irldta', 'ivolts',
        'ixyz', 'imb', 'ims', 'iml', 'imw',
        'imd', 'imx', 'ineic', 'ipde', 'iisc',
        'ireb', 'iusgs', 'ibrk', 'icaltech', 'illnl',
        'ievloc', 'ijsop', 'iuser', 'iunknown', 'iqb',
        'iqb1', 'iqb2', 'iqbx', 'iqmt', 'ieq',
        'ieq1', 'ieq2', 'ime', 'iex', 'inu',
        'inc', 'io', 'il', 'ir', 'it',
        'iu',
    )

    def __init__(self, name, eol=False, default=None):
        self.name = name
        self._name = '_' + self.name
        self.type = self._type(self.name)
        self.eol = eol
        self.default = default
        self.value_from_internal = self._value_from_internal(self.type)
        self.internal_from_value = self._internal_from_value(self.type)
        self.parse_ascii = self.PARSE_ASCII_FROM_TYPE[self.type]
        self.ascii_format = self._ascii_format(self.type, eol)
        self.n_ascii_bytes = self.ASCII_BYTES_FROM_TYPE[self.type]
        self.binary_format = self.BINARY_FORMAT_FROM_TYPE[self.type]

    def to_ascii(self, x):
        return self.ascii_format.format(x)

    @classmethod
    def _value_from_internal(cls, t):
        return getattr(cls, '_{}_from_internal'.format(t))

    @classmethod
    def _internal_from_value(cls, t):
        return getattr(cls, '_internal_from_{}'.format(t))

    @staticmethod
    @_none_from_undefined_1(UNDEFINED_LOGICAL)
    def _logical_from_internal(n):
        return n == 1

    @staticmethod
    @_undefined_from_none_1(UNDEFINED_LOGICAL)
    def _internal_from_logical(l):
        if l:
            return 1
        else:
            return 0

    @staticmethod
    @_none_from_undefined_1(UNDEFINED_INTEGER)
    def _integer_from_internal(n):
        return n

    @staticmethod
    @_undefined_from_none_1(UNDEFINED_INTEGER)
    @_assert_range_1(INTEGER_MIN, INTEGER_MAX)
    def _internal_from_integer(n):
        return n

    @staticmethod
    @_none_from_undefined_1(UNDEFINED_FLOAT)
    def _float_from_internal(x):
        return x

    @staticmethod
    @_undefined_from_none_1(UNDEFINED_FLOAT)
    @_assert_range_1(FLOAT_MIN, FLOAT_MAX)
    def _internal_from_float(x):
        return x

    @classmethod
    @_none_from_undefined_2(UNDEFINED_ENUM)
    @_assert_range_2(1, len(ENUMS))
    def _enum_from_internal(cls, n):
        return cls.ENUMS[n - 1]

    @classmethod
    @_undefined_from_none_2(UNDEFINED_ENUM)
    def _internal_from_enum(cls, s):
        return cls.ENUMS.index(s) + 1

    @staticmethod
    @_none_from_undefined_1(UNDEFINED_SHORT_STRING)
    def _short_string_from_internal(s):
        return s.rstrip()

    @classmethod
    @_undefined_from_none_2(UNDEFINED_SHORT_STRING)
    def _internal_from_short_string(cls, s):
        return cls._pad_space(s, cls.BYTES_SHORT_STRING)

    @staticmethod
    @_none_from_undefined_1(UNDEFINED_LONG_STRING)
    def _long_string_from_internal(s):
        return s.rstrip()

    @classmethod
    @_undefined_from_none_2(UNDEFINED_LONG_STRING)
    def _internal_from_long_string(cls, s):
        return cls._pad_space(s, cls.BYTES_LONG_STRING)

    @staticmethod
    def _pad_space(s, n):
        bs = s.encode()
        nbs = len(bs)
        assert nbs <= n
        return (bs + b' '*(n - nbs)).decode()

    @staticmethod
    def _type(name):
        prefix = name[0]
        if prefix == 'n':
            return 'integer'
        elif prefix == 'i':
            return 'enum'
        elif prefix == 'l':
            return 'logical'
        elif prefix == 'k':
            if name == 'kevnm':
                return 'long_string'
            else:
                return 'short_string'
        else:
            return 'float'

    @classmethod
    def _ascii_format(cls, type, eol):
        ret = cls.ASCII_FORMAT_FROM_TYPE[type]
        if eol:
            ret += '\n'
        return ret


class Meta(object):

    FIELDS = (
        FieldProp(name='delta'),
        FieldProp(name='depmin'),
        FieldProp(name='depmax'),
        FieldProp(name='scale'),
        FieldProp(name='odelta', eol=True),
        FieldProp(name='b'),
        FieldProp(name='e'),
        FieldProp(name='o'),
        FieldProp(name='a'),
        FieldProp(name='fmt', eol=True),
        FieldProp(name='t0'),
        FieldProp(name='t1'),
        FieldProp(name='t2'),
        FieldProp(name='t3'),
        FieldProp(name='t4', eol=True),
        FieldProp(name='t5'),
        FieldProp(name='t6'),
        FieldProp(name='t7'),
        FieldProp(name='t8'),
        FieldProp(name='t9', eol=True),
        FieldProp(name='f'),
        FieldProp(name='resp0'),
        FieldProp(name='resp1'),
        FieldProp(name='resp2'),
        FieldProp(name='resp3', eol=True),
        FieldProp(name='resp4'),
        FieldProp(name='resp5'),
        FieldProp(name='resp6'),
        FieldProp(name='resp7'),
        FieldProp(name='resp8', eol=True),
        FieldProp(name='resp9'),
        FieldProp(name='stla'),
        FieldProp(name='stlo'),
        FieldProp(name='stel'),
        FieldProp(name='stdp', eol=True),
        FieldProp(name='evla'),
        FieldProp(name='evlo'),
        FieldProp(name='evel'),
        FieldProp(name='evdp'),
        FieldProp(name='mag', eol=True),
        FieldProp(name='user0'),
        FieldProp(name='user1'),
        FieldProp(name='user2'),
        FieldProp(name='user3'),
        FieldProp(name='user4', eol=True),
        FieldProp(name='user5'),
        FieldProp(name='user6'),
        FieldProp(name='user7'),
        FieldProp(name='user8'),
        FieldProp(name='user9', eol=True),
        FieldProp(name='dist'),
        FieldProp(name='az'),
        FieldProp(name='baz'),
        FieldProp(name='gcarc'),
        FieldProp(name='sb', eol=True),
        FieldProp(name='sdelta'),
        FieldProp(name='depmen'),
        FieldProp(name='cmpaz'),
        FieldProp(name='cmpinc'),
        FieldProp(name='xminimum', eol=True),
        FieldProp(name='xmaximum'),
        FieldProp(name='yminimum'),
        FieldProp(name='ymaximum'),
        FieldProp(name='fhdr64'), # adjtm
        FieldProp(name='fhdr65', eol=True),
        FieldProp(name='fhdr66'),
        FieldProp(name='fhdr67'),
        FieldProp(name='fhdr68'),
        FieldProp(name='fhdr69'),
        FieldProp(name='fhdr70', eol=True),
        FieldProp(name='nzyear'),
        FieldProp(name='nzjday'),
        FieldProp(name='nzhour'),
        FieldProp(name='nzmin'),
        FieldProp(name='nzsec', eol=True),
        FieldProp(name='nzmsec'),
        FieldProp(name='nvhdr'),
        FieldProp(name='norid'),
        FieldProp(name='nevid'),
        FieldProp(name='npts', eol=True),
        FieldProp(name='nspts'),
        FieldProp(name='nwfid'),
        FieldProp(name='nxsize'),
        FieldProp(name='nysize'),
        FieldProp(name='nhdr56', eol=True),
        FieldProp(name='iftype'),
        FieldProp(name='idep'),
        FieldProp(name='iztype'),
        FieldProp(name='ihdr4'),
        FieldProp(name='iinst', eol=True),
        FieldProp(name='istreg'),
        FieldProp(name='ievreg'),
        FieldProp(name='ievtyp'),
        FieldProp(name='iqual'),
        FieldProp(name='isynth', eol=True),
        FieldProp(name='imagtyp'),
        FieldProp(name='imagsrc'),
        FieldProp(name='ihdr13'),
        FieldProp(name='ihdr14'),
        FieldProp(name='ihdr15', eol=True),
        FieldProp(name='ihdr16'),
        FieldProp(name='ihdr17'),
        FieldProp(name='ihdr18'),
        FieldProp(name='ihdr19'),
        FieldProp(name='ihdr20', eol=True),
        FieldProp(name='leven'),
        FieldProp(name='lpspol'),
        FieldProp(name='lovrok'),
        FieldProp(name='lcalda'),
        FieldProp(name='lhdr5', eol=True),
        FieldProp(name='kstnm'),
        FieldProp(name='kevnm', eol=True),
        FieldProp(name='khole'),
        FieldProp(name='ko'),
        FieldProp(name='ka', eol=True),
        FieldProp(name='kt0'),
        FieldProp(name='kt1'),
        FieldProp(name='kt2', eol=True),
        FieldProp(name='kt3'),
        FieldProp(name='kt4'),
        FieldProp(name='kt5', eol=True),
        FieldProp(name='kt6'),
        FieldProp(name='kt7'),
        FieldProp(name='kt8', eol=True),
        FieldProp(name='kt9'),
        FieldProp(name='kf'),
        FieldProp(name='kuser0', eol=True),
        FieldProp(name='kuser1'),
        FieldProp(name='kuser2'),
        FieldProp(name='kcmpnm', eol=True),
        FieldProp(name='knetwk'),
        FieldProp(name='kdatrd'),
        FieldProp(name='kinst', eol=True),
    )

    NAMES = tuple(f.name for f in FIELDS)
    BINARY_FORMAT = ''.join(field.binary_format for field in FIELDS)

    def __init__(self):
        for field in self.FIELDS:
            setattr(self, field.name, field.default)

    def from_dict(self, d):
        self.from_list(d[name] for name in self.NAMES)
        return self

    def to_dict(self):
        return {name: getattr(self, name) for name in self.NAMES}

    def from_list(self, vs):
        assert len(vs) == len(self.NAMES)
        for name, value in zip(self.NAMES, vs):
            setattr(self, name, value)
        return self

    def to_ascii(self):
        return ''.join(field.to_ascii(getattr(self, field._name)) for field in self.FIELDS)

    def from_binary(self, b):
        return self._from_meta_list(struct.unpack(self.BINARY_FORMAT, b))

    def to_binary(self):
        vs = []
        for field in self.FIELDS:
            t = field.type
            v = getattr(self, field._name)
            if t == 'short_string' or t == 'long_string':
                bytes_ = v.encode()
                for i in range(len(bytes_)):
                    vs.append(bytes_[i:i+1])
            else:
                vs.append(v)
        return struct.pack(self.BINARY_FORMAT, *vs)

    def _from_meta_list(self, vs):
        iv = 0
        for field in self.FIELDS:
            t = field.type
            _name = field._name
            if t == 'short_string' or t == 'long_string':
                iv_next = iv + self.INTERNAL_BYTES_FROM_TYPE[t]
                setattr(self, _name, b''.join(vs[iv:iv_next]).decode('utf-8'))
                iv = iv_next
            else:
                setattr(self, _name, vs[iv])
                iv += 1
        return self

    @classmethod
    def _make_from_ascii(cls):
        _name_il_ir_fns = []
        il = 0
        for field in cls.FIELDS:
            ir = il + field.n_ascii_bytes
            _name_il_ir_fns.append((field._name, il, ir, field.parse_ascii))
            il = ir
            if field.eol:
                ir = il + 1
                il = ir
        ascii_byte = ir

        def from_ascii(self, s):
            bs = s.encode()
            assert len(bs) == ascii_byte
            for _name, il, ir, fn in _name_il_ir_fns:
                setattr(self, _name, fn(bs[il:ir].decode()))
            return self
        return from_ascii

    @staticmethod
    def _make_property(field):
        return property(lambda self: field.value_from_internal(getattr(self, field._name)),
                        lambda self, value: setattr(self, field._name, field.internal_from_value(value)))

Meta.from_ascii = Meta._make_from_ascii()

for field in Meta.FIELDS:
    setattr(Meta, field.name, Meta._make_property(field))
# `adjtm` seems not used
# Meta.adjtm = property(lambda self: self._float_from_internal(self._fhdr64),
#                       lambda self, value: setattr(self, '_fhdr64', self._internal_from_float(value)))


class Data(object):
    pass


class Sac(object):
    def __init__(self):
        self._meta = Meta()
        self._data = Data()
    pass

if __name__ == '__main__':
    import unittest
    from random import randint


    class Tester(unittest.TestCase):

        def setUp(self):
            self.h = Meta()

        def test_ascii(self):
            self.assertEqual(self.h.to_ascii(), Meta().from_ascii(self.h.to_ascii()).to_ascii())
            s = """\
      -12345.00      -12345.00      -12345.00      -12345.00      -12345.00
      -12345.00      -12345.00      -12345.00      -12345.00      -12345.00
      -12345.00      -12345.00      -12345.00      -12345.00      -12345.00
      -12345.00      -12345.00      -12345.00      -12345.00      -12345.00
      -12345.00      -12345.00      -12345.00      -12345.00      -12345.00
      -12345.00      -12345.00      -12345.00      -12345.00      -12345.00
      -12345.00      -12345.00      -12345.00      -12345.00      -12345.00
      -12345.00      -12345.00      -12345.00      -12345.00      -12345.00
      -12345.00      -12345.00      -12345.00      -12345.00      -12345.00
      -12345.00      -12345.00      -12345.00      -12345.00      -12345.00
      -12345.00      -12345.00      -12345.00      -12345.00      -12345.00
      -12345.00      -12345.00      -12345.00      -12345.00      -12345.00
      -12345.00      -12345.00      -12345.00      -12345.00      -12345.00
      -12345.00      -12345.00      -12345.00      -12345.00      -12345.00
    -12345    -12345    -12345    -12345    -12345
    -12345    -12345    -12345    -12345    -12345
    -12345    -12345    -12345    -12345    -12345
    -12345    -12345    -12345    -12345    -12345
    -12345    -12345    -12345    -12345    -12345
    -12345    -12345    -12345    -12345    -12345
    -12345    -12345    -12345    -12345    -12345
    -12345    -12345    -12345    -12345    -12345
-12345  -12345          
-12345  -12345  -12345  
-12345  -12345  -12345  
-12345  -12345  -12345  
-12345  -12345  -12345  
-12345  -12345  -12345  
-12345  -12345  -12345  
-12345  -12345  -12345  
"""
            self.assertEqual(self.h.to_ascii(), s)
            with self.assertRaises(AssertionError):
                self.h.from_ascii('too short')

        def test_internal_converters(self):
            self.h.delta = 1.0
            self.assertAlmostEqual(self.h._delta, 1.0)
            self.assertAlmostEqual(self.h.delta, 1.0)
            self.h.delta = None
            self.assertAlmostEqual(self.h._delta, FieldProp.UNDEFINED_FLOAT)
            self.assertTrue(self.h.delta is None)
            with self.assertRaises(AssertionError):
                self.h.delta = 2*FLOAT_MIN
            with self.assertRaises(AssertionError):
                self.h.delta = 2*FLOAT_MAX

            self.h.nzyear = 2
            self.assertEqual(self.h._nzyear, 2)
            self.assertEqual(self.h.nzyear, 2)
            self.h.nzyear = None
            self.assertEqual(self.h._nzyear, FieldProp.UNDEFINED_INTEGER)
            self.assertEqual(self.h.nzyear, None)
            with self.assertRaises(AssertionError):
                self.h.nzyear = 2*INTEGER_MIN
            with self.assertRaises(AssertionError):
                self.h.nzyear = 2*FLOAT_MAX

            self.h.iftype = 'it'
            self.assertEqual(self.h._iftype, 85)
            self.assertEqual(self.h.iftype, 'it')
            self.h.iftype = None
            self.assertEqual(self.h._iftype, FieldProp.UNDEFINED_ENUM)
            self.assertEqual(self.h.iftype, None)
            with self.assertRaises(Exception):
                self.h.iftype = 'not_member_of_enums'
            self.h._iftype = 0
            with self.assertRaises(AssertionError):
                self.h.iftype
            self.h._iftype = len(FieldProp.ENUMS) + 1
            with self.assertRaises(Exception):
                self.h.iftype

            self.h.leven = True
            self.assertEqual(self.h._leven, 1)
            self.assertEqual(self.h.leven, True)
            self.h.leven = False
            self.assertEqual(self.h._leven, 0)
            self.assertEqual(self.h.leven, False)
            self.h.leven = None
            self.assertEqual(self.h._leven, FieldProp.UNDEFINED_LOGICAL)
            self.assertEqual(self.h.leven, None)

            self.h.kstnm = 'erm'
            self.assertEqual(self.h._kstnm, 'erm     ')
            self.assertEqual(self.h.kstnm, 'erm')
            self.h.kstnm = '筑波'
            self.assertEqual(self.h._kstnm, '筑波  ')
            self.assertEqual(self.h.kstnm, '筑波')
            self.h.kstnm = ''
            self.assertEqual(self.h._kstnm, '        ')
            self.assertEqual(self.h.kstnm, '')
            self.h.kstnm = None
            self.assertEqual(self.h._kstnm, FieldProp.UNDEFINED_SHORT_STRING)
            self.assertEqual(self.h.kstnm, None)
            with self.assertRaises(AssertionError):
                self.h.kstnm = '123456789'

            self.h.kevnm = 'sanriku-oki'
            self.assertEqual(self.h._kevnm, 'sanriku-oki     ')
            self.assertEqual(self.h.kevnm, 'sanriku-oki')
            self.h.kevnm = ''
            self.assertEqual(self.h._kevnm, ' '*16)
            self.assertEqual(self.h.kevnm, '')
            self.h.kevnm = None
            self.assertEqual(self.h._kevnm, FieldProp.UNDEFINED_LONG_STRING)
            self.assertEqual(self.h.kevnm, None)
            with self.assertRaises(AssertionError):
                self.h.kevnm = '0123456789abcdefg'

    unittest.main()
