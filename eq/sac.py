"""
# REFERENCES
- [SAC User Manual/SAC Data Format](http://www.iris.edu/software/sac/manual/file_format.html)
"""

import struct as _struct


_FLOAT = _pylab.float32
_INTEGER = _pylab.int32
_COMPLEX = _pylab.complex64


_INTEGER_MIN = _pylab.iinfo(_INTEGER).min
_INTEGER_MAX = _pylab.iinfo(_INTEGER).max
_FLOAT_MIN = _pylab.finfo(_FLOAT).min
_FLOAT_MAX = _pylab.finfo(_FLOAT).max


_INTERNAL_BYTES_FROM_TYPE = {
    'short_string': 8,
    'long_string': 16,
}
_INTERNAL_BYTES_SHORT_STRING = _INTERNAL_BYTES_FROM_TYPE['short_string']
_INTERNAL_BYTES_LONG_STRING = _INTERNAL_BYTES_FROM_TYPE['long_string']

_ASCII_BYTES_FROM_TYPE = {
    'logical': 10,
    'integer': 10,
    'float': 15,
    'enum': 10,
    'short_string': _INTERNAL_BYTES_SHORT_STRING,
    'long_string': _INTERNAL_BYTES_LONG_STRING,
}
_ASCII_BYTES_SHORT_STRING = _ASCII_BYTES_FROM_TYPE['short_string']
_ASCII_BYTES_LONG_STRING = _ASCII_BYTES_FROM_TYPE['long_string']

_ASCII_FORMAT_FROM_TYPE = {
    'logical': '{{:>{}d}}'.format(_ASCII_BYTES_FROM_TYPE['logical']),
    'integer': '{{:>{}d}}'.format(_ASCII_BYTES_FROM_TYPE['integer']),
    'float': '{{:>#{}.7g}}'.format(_ASCII_BYTES_FROM_TYPE['float']),
    'enum': '{{:>{}d}}'.format(_ASCII_BYTES_FROM_TYPE['enum']),
    'short_string': '{:s}', # assume internal values to always have BYTES_SHORT_STRING bytes
    'long_string': '{:s}', # assume internal values to always have BYTES_LONG_STRING bytes
}
_PARSE_ASCII_FROM_TYPE = {
    'logical': int,
    'integer': int,
    'float': float,
    'enum': int,
    'short_string': lambda s: s,
    'long_string': lambda s: s,
}
_BINARY_FORMAT_FROM_TYPE = {
    'logical': 'i',
    'integer': 'i',
    'float': 'f',
    'enum': 'i',
    'short_string': '{}c'.format(_ASCII_BYTES_SHORT_STRING),
    'long_string': '{}c'.format(_ASCII_BYTES_LONG_STRING),
}


_UNDEFINED_FROM_TYPE = {
    'logical': -12345,
    'integer': -12345,
    'float': -12345.0,
    'enum': -12345,
    'short_string': '-12345  ',
    'long_string': '-12345          ',
}
_UNDEFINED_LOGICAL = _UNDEFINED_FROM_TYPE['logical']
_UNDEFINED_INTEGER = _UNDEFINED_FROM_TYPE['integer']
_UNDEFINED_FLOAT = _UNDEFINED_FROM_TYPE['float']
_UNDEFINED_ENUM = _UNDEFINED_FROM_TYPE['enum']
_UNDEFINED_SHORT_STRING = _UNDEFINED_FROM_TYPE['short_string']
_UNDEFINED_LONG_STRING = _UNDEFINED_FROM_TYPE['long_string']


_ENUMS = (
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


def _convert_from(b, a):
    def wrapper(f):
        def new_f(x):
            if x == a:
                return b
            else:
                return f(x)
        return new_f
    return wrapper
_none_from_undefined = lambda undefined: _convert_from(None, undefined)
_undefined_from_none = lambda undefined: _convert_from(undefined, None)


def _assert_range(lower, upper):
    def wrapper(f):
        def new_f(x):
            assert lower <= x <= upper
            return f(x)
        return new_f
    return wrapper


def _pad_space(s, n):
    bs = s.encode()
    nbs = len(bs)
    assert nbs <= n
    return (bs + b' '*(n - nbs)).decode()


class _FieldProp(object):

    def __init__(self, name, eol=False, default=None):
        self.name = name
        self._name = '_' + self.name
        self.type = self._type(self.name)
        self.eol = eol
        self.default = default
        self.value_from_internal = self._value_from_internal(self.type)
        self.internal_from_value = self._internal_from_value(self.type)
        self.parse_ascii = _PARSE_ASCII_FROM_TYPE[self.type]
        self.ascii_format = self._ascii_format(self.type, eol)
        self.n_ascii_bytes = _ASCII_BYTES_FROM_TYPE[self.type]
        self.binary_format = _BINARY_FORMAT_FROM_TYPE[self.type]

    def to_ascii(self, x):
        return self.ascii_format.format(x)

    @classmethod
    def _value_from_internal(cls, t):
        return getattr(cls, '_{}_from_internal'.format(t))

    @classmethod
    def _internal_from_value(cls, t):
        return getattr(cls, '_internal_from_{}'.format(t))

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
        ret = _ASCII_FORMAT_FROM_TYPE[type]
        if eol:
            ret += '\n'
        return ret

for _t, _from_internal, _to_internal in (('logical',
                                          lambda n: n == 1,
                                          lambda l: 1 if l else 0),
                                         ('integer',
                                          lambda n: n,
                                          _assert_range(_INTEGER_MIN, _INTEGER_MAX)(lambda n: n)),
                                         ('float',
                                          lambda x: x,
                                          _assert_range(_FLOAT_MIN, _FLOAT_MAX)(lambda x: x)),
                                         ('enum',
                                          _assert_range(1, len(_ENUMS))(lambda n: _ENUMS[n - 1]),
                                          lambda s: _ENUMS.index(s) + 1),
                                         ('short_string',
                                          lambda s: s.rstrip(),
                                          lambda s: _pad_space(s, _INTERNAL_BYTES_SHORT_STRING)),
                                         ('long_string',
                                          lambda s: s.rstrip(),
                                          lambda s: _pad_space(s, _INTERNAL_BYTES_LONG_STRING))):
    setattr(_FieldProp,
            '_{}_from_internal'.format(_t),
            staticmethod(_none_from_undefined(_UNDEFINED_FROM_TYPE[_t])(_from_internal)))
    setattr(_FieldProp,
            '_internal_from_{}'.format(_t),
            staticmethod(_undefined_from_none(_UNDEFINED_FROM_TYPE[_t])(_to_internal)))


class _Meta(object):

    FIELDS = (
        _FieldProp(name='delta'),
        _FieldProp(name='depmin'),
        _FieldProp(name='depmax'),
        _FieldProp(name='scale'),
        _FieldProp(name='odelta', eol=True),
        _FieldProp(name='b'),
        _FieldProp(name='e'),
        _FieldProp(name='o'),
        _FieldProp(name='a'),
        _FieldProp(name='fmt', eol=True),
        _FieldProp(name='t0'),
        _FieldProp(name='t1'),
        _FieldProp(name='t2'),
        _FieldProp(name='t3'),
        _FieldProp(name='t4', eol=True),
        _FieldProp(name='t5'),
        _FieldProp(name='t6'),
        _FieldProp(name='t7'),
        _FieldProp(name='t8'),
        _FieldProp(name='t9', eol=True),
        _FieldProp(name='f'),
        _FieldProp(name='resp0'),
        _FieldProp(name='resp1'),
        _FieldProp(name='resp2'),
        _FieldProp(name='resp3', eol=True),
        _FieldProp(name='resp4'),
        _FieldProp(name='resp5'),
        _FieldProp(name='resp6'),
        _FieldProp(name='resp7'),
        _FieldProp(name='resp8', eol=True),
        _FieldProp(name='resp9'),
        _FieldProp(name='stla'),
        _FieldProp(name='stlo'),
        _FieldProp(name='stel'),
        _FieldProp(name='stdp', eol=True),
        _FieldProp(name='evla'),
        _FieldProp(name='evlo'),
        _FieldProp(name='evel'),
        _FieldProp(name='evdp'),
        _FieldProp(name='mag', eol=True),
        _FieldProp(name='user0'),
        _FieldProp(name='user1'),
        _FieldProp(name='user2'),
        _FieldProp(name='user3'),
        _FieldProp(name='user4', eol=True),
        _FieldProp(name='user5'),
        _FieldProp(name='user6'),
        _FieldProp(name='user7'),
        _FieldProp(name='user8'),
        _FieldProp(name='user9', eol=True),
        _FieldProp(name='dist'),
        _FieldProp(name='az'),
        _FieldProp(name='baz'),
        _FieldProp(name='gcarc'),
        _FieldProp(name='sb', eol=True),
        _FieldProp(name='sdelta'),
        _FieldProp(name='depmen'),
        _FieldProp(name='cmpaz'),
        _FieldProp(name='cmpinc'),
        _FieldProp(name='xminimum', eol=True),
        _FieldProp(name='xmaximum'),
        _FieldProp(name='yminimum'),
        _FieldProp(name='ymaximum'),
        _FieldProp(name='fhdr64'), # adjtm
        _FieldProp(name='fhdr65', eol=True),
        _FieldProp(name='fhdr66'),
        _FieldProp(name='fhdr67'),
        _FieldProp(name='fhdr68'),
        _FieldProp(name='fhdr69'),
        _FieldProp(name='fhdr70', eol=True),
        _FieldProp(name='nzyear'),
        _FieldProp(name='nzjday'),
        _FieldProp(name='nzhour'),
        _FieldProp(name='nzmin'),
        _FieldProp(name='nzsec', eol=True),
        _FieldProp(name='nzmsec'),
        _FieldProp(name='nvhdr'),
        _FieldProp(name='norid'),
        _FieldProp(name='nevid'),
        _FieldProp(name='npts', eol=True),
        _FieldProp(name='nspts'),
        _FieldProp(name='nwfid'),
        _FieldProp(name='nxsize'),
        _FieldProp(name='nysize'),
        _FieldProp(name='nhdr56', eol=True),
        _FieldProp(name='iftype'),
        _FieldProp(name='idep'),
        _FieldProp(name='iztype'),
        _FieldProp(name='ihdr4'),
        _FieldProp(name='iinst', eol=True),
        _FieldProp(name='istreg'),
        _FieldProp(name='ievreg'),
        _FieldProp(name='ievtyp'),
        _FieldProp(name='iqual'),
        _FieldProp(name='isynth', eol=True),
        _FieldProp(name='imagtyp'),
        _FieldProp(name='imagsrc'),
        _FieldProp(name='ihdr13'),
        _FieldProp(name='ihdr14'),
        _FieldProp(name='ihdr15', eol=True),
        _FieldProp(name='ihdr16'),
        _FieldProp(name='ihdr17'),
        _FieldProp(name='ihdr18'),
        _FieldProp(name='ihdr19'),
        _FieldProp(name='ihdr20', eol=True),
        _FieldProp(name='leven'),
        _FieldProp(name='lpspol'),
        _FieldProp(name='lovrok'),
        _FieldProp(name='lcalda'),
        _FieldProp(name='lhdr5', eol=True),
        _FieldProp(name='kstnm'),
        _FieldProp(name='kevnm', eol=True),
        _FieldProp(name='khole'),
        _FieldProp(name='ko'),
        _FieldProp(name='ka', eol=True),
        _FieldProp(name='kt0'),
        _FieldProp(name='kt1'),
        _FieldProp(name='kt2', eol=True),
        _FieldProp(name='kt3'),
        _FieldProp(name='kt4'),
        _FieldProp(name='kt5', eol=True),
        _FieldProp(name='kt6'),
        _FieldProp(name='kt7'),
        _FieldProp(name='kt8', eol=True),
        _FieldProp(name='kt9'),
        _FieldProp(name='kf'),
        _FieldProp(name='kuser0', eol=True),
        _FieldProp(name='kuser1'),
        _FieldProp(name='kuser2'),
        _FieldProp(name='kcmpnm', eol=True),
        _FieldProp(name='knetwk'),
        _FieldProp(name='kdatrd'),
        _FieldProp(name='kinst', eol=True),
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
        return self._from_meta_list(_struct.unpack(self.BINARY_FORMAT, b))

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
        return _struct.pack(self.BINARY_FORMAT, *vs)

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

_Meta.from_ascii = _Meta._make_from_ascii()

for _field in _Meta.FIELDS:
    setattr(_Meta, _field.name, _Meta._make_property(_field))
# `adjtm` seems not used
# _Meta.adjtm = property(lambda self: self._float_from_internal(self._fhdr64),
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
            self.h = _Meta()

        def test_ascii(self):
            self.assertEqual(self.h.to_ascii(), _Meta().from_ascii(self.h.to_ascii()).to_ascii())
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
            self.assertAlmostEqual(self.h._delta, _UNDEFINED_FLOAT)
            self.assertTrue(self.h.delta is None)
            with self.assertRaises(AssertionError):
                self.h.delta = 2*_FLOAT_MIN
            with self.assertRaises(AssertionError):
                self.h.delta = 2*_FLOAT_MAX

            self.h.nzyear = 2
            self.assertEqual(self.h._nzyear, 2)
            self.assertEqual(self.h.nzyear, 2)
            self.h.nzyear = None
            self.assertEqual(self.h._nzyear, _UNDEFINED_INTEGER)
            self.assertEqual(self.h.nzyear, None)
            with self.assertRaises(AssertionError):
                self.h.nzyear = 2*_INTEGER_MIN
            with self.assertRaises(AssertionError):
                self.h.nzyear = 2*_FLOAT_MAX

            self.h.iftype = 'it'
            self.assertEqual(self.h._iftype, 85)
            self.assertEqual(self.h.iftype, 'it')
            self.h.iftype = None
            self.assertEqual(self.h._iftype, _UNDEFINED_ENUM)
            self.assertEqual(self.h.iftype, None)
            with self.assertRaises(Exception):
                self.h.iftype = 'not_member_of_enums'
            self.h._iftype = 0
            with self.assertRaises(AssertionError):
                self.h.iftype
            self.h._iftype = len(_ENUMS) + 1
            with self.assertRaises(Exception):
                self.h.iftype

            self.h.leven = True
            self.assertEqual(self.h._leven, 1)
            self.assertEqual(self.h.leven, True)
            self.h.leven = False
            self.assertEqual(self.h._leven, 0)
            self.assertEqual(self.h.leven, False)
            self.h.leven = None
            self.assertEqual(self.h._leven, _UNDEFINED_LOGICAL)
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
            self.assertEqual(self.h._kstnm, _UNDEFINED_SHORT_STRING)
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
            self.assertEqual(self.h._kevnm, _UNDEFINED_LONG_STRING)
            self.assertEqual(self.h.kevnm, None)
            with self.assertRaises(AssertionError):
                self.h.kevnm = '0123456789abcdefg'

    unittest.main()
