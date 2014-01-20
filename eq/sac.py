"""
# REFERENCES
- [SAC User Manual/SAC Data Format](http://www.iris.edu/software/sac/manual/file_format.html)
"""

import struct as _struct

import pylab as _pylab

class Error(Exception):
    pass


def error(cond, msg):
    if not cond:
        raise Error(msg)


_FLOAT = _pylab.float32
_INTEGER = _pylab.int32
_COMPLEX = _pylab.complex64


_INTEGER_MIN = _pylab.iinfo(_INTEGER).min
_INTEGER_MAX = _pylab.iinfo(_INTEGER).max
_FLOAT_MIN = _pylab.finfo(_FLOAT).min
_FLOAT_MAX = _pylab.finfo(_FLOAT).max

_BINARY_MODE = '='

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
    'short_string': '{}s'.format(_ASCII_BYTES_SHORT_STRING),
    'long_string': '{}s'.format(_ASCII_BYTES_LONG_STRING),
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


def from_(x):
    return Sac().from_(x)


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


def _assert_floats(xs):
    assert _pylab.all(_FLOAT_MIN <= xs) and _pylab.all(xs <= _FLOAT_MAX)
    return xs

def _identity(x):
    return x


class _FieldProp(object):

    def __init__(self, name, eol=False, default=None, type_=None):
        self.name = name
        self._name = '_' + self.name
        if type_ is None:
            self.type_ = self._type(self.name)
        else:
            self.type_ = type_
        self.eol = eol
        self.default = default
        self.value_from_internal = getattr(self, '_{}_value_from_internal'.format(self.type_))
        self.internal_from_value = getattr(self, '_internal_from_{}_value'.format(self.type_))
        self.pack_value_from_internal = getattr(self, '_{}_pack_value_from_internal'.format(self.type_))
        self.internal_from_unpacked = getattr(self, '_internal_from_unpacked_{}'.format(self.type_))
        self.parse_ascii = _PARSE_ASCII_FROM_TYPE[self.type_]
        self.ascii_format = self._ascii_format(self.type_, eol)
        self.n_ascii_bytes = _ASCII_BYTES_FROM_TYPE[self.type_]
        self.binary_format = _BINARY_FORMAT_FROM_TYPE[self.type_]

    def to_ascii(self, x):
        return self.ascii_format.format(x)

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
            return 'short_string'
        else:
            return 'float'

    @classmethod
    def _ascii_format(cls, type, eol):
        ret = _ASCII_FORMAT_FROM_TYPE[type]
        if eol:
            ret += '\n'
        return ret


for (_t,
     _value_from_internal,
     _internal_from_value,
     _pack_value_from_internal,
     _internal_from_unpacked) in (('logical',
                                   lambda n: n == 1,
                                   lambda l: 1 if l else 0,
                                   _identity,
                                   _identity),
                                  ('integer',
                                   _identity,
                                   _assert_range(_INTEGER_MIN, _INTEGER_MAX)(_identity),
                                   _identity,
                                   _identity),
                                  ('float',
                                   _identity,
                                   _assert_range(_FLOAT_MIN, _FLOAT_MAX)(_identity),
                                   _identity,
                                   _identity),
                                  ('enum',
                                   _assert_range(1, len(_ENUMS))(lambda n: _ENUMS[n - 1]),
                                   lambda s: _ENUMS.index(s) + 1,
                                   _identity,
                                   _identity),
                                  ('short_string',
                                   lambda s: s.rstrip(),
                                   lambda s: _pad_space(s, _INTERNAL_BYTES_SHORT_STRING),
                                   lambda s: s.encode(),
                                   lambda b: b.decode()),
                                  ('long_string',
                                   lambda s: s.rstrip(),
                                   lambda s: _pad_space(s, _INTERNAL_BYTES_LONG_STRING),
                                   lambda s: s.encode(),
                                   lambda b: b.decode())):
    setattr(_FieldProp,
            '_{}_value_from_internal'.format(_t),
            staticmethod(_none_from_undefined(_UNDEFINED_FROM_TYPE[_t])(_value_from_internal)))
    setattr(_FieldProp,
            '_internal_from_{}_value'.format(_t),
            staticmethod(_undefined_from_none(_UNDEFINED_FROM_TYPE[_t])(_internal_from_value)))
    setattr(_FieldProp,
            '_{}_pack_value_from_internal'.format(_t),
            staticmethod(_pack_value_from_internal))
    setattr(_FieldProp,
            '_internal_from_unpacked_{}'.format(_t),
            staticmethod(_internal_from_unpacked))


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
        _FieldProp(name='nvhdr', default=6),
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
        _FieldProp(name='lovrok', default=True),
        _FieldProp(name='lcalda'),
        _FieldProp(name='lhdr5', eol=True),
        _FieldProp(name='kstnm'),
        _FieldProp(name='kevnm', eol=True, type_='long_string'),
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
    BINARY_FORMAT = _BINARY_MODE + ''.join(field.binary_format for field in FIELDS)
    BYTES_BINARY_FORMAT = _struct.calcsize(BINARY_FORMAT)
    BYTES_ASCII_FORMAT = sum(_ASCII_BYTES_FROM_TYPE[field.type_] + (1 if field.eol else 0)
                             for field
                             in FIELDS)

    def __init__(self):
        for field in self.FIELDS:
            setattr(self, field.name, field.default)

    def from_dict(self, d):
        for field in self.FIELDS:
            name = field.name
            setattr(self, name, d[name])
        return self

    def to_dict(self):
        return {name: getattr(self, name) for name in self.NAMES}

    def __str__(self):
        return ''.join(field.to_ascii(getattr(self, field._name)) for field in self.FIELDS)

    def from_bytes(self, b):
        for field, value in zip(self.FIELDS, _struct.unpack(self.BINARY_FORMAT, b)):
            setattr(self, field._name, self.internal_from_unpacked(value))
        return self

    def __bytes__(self):
        return _struct.pack(self.BINARY_FORMAT,
                            *(field.pack_value_from_internal(getattr(self, field._name))
                              for field
                              in self.FIELDS))

    @classmethod
    def _make_from_str(cls):
        _name_il_ir_fns = []
        il = 0
        for field in cls.FIELDS:
            ir = il + field.n_ascii_bytes
            _name_il_ir_fns.append((field._name, il, ir, field.parse_ascii))
            il = ir
            if field.eol:
                ir = il + 1
                il = ir
        n_ascii_bytes = ir

        def from_str(self, s):
            bs = s.encode()
            assert len(bs) == n_ascii_bytes
            for _name, il, ir, fn in _name_il_ir_fns:
                setattr(self, _name, fn(bs[il:ir].decode()))
            return self
        return from_str

    @staticmethod
    def _make_property(field):
        return property(lambda self: field.value_from_internal(getattr(self, field._name)),
                        lambda self, value: setattr(self, field._name, field.internal_from_value(value)))

_Meta.from_str = _Meta._make_from_str()

for _field in _Meta.FIELDS:
    setattr(_Meta, _field.name, _Meta._make_property(_field))
# `adjtm` seems not used
# _Meta.adjtm = property(lambda self: self._float_from_internal(self._fhdr64),
#                       lambda self, value: setattr(self, '_fhdr64', self._internal_from_float(value)))


class Sac(object):

    BINARY_FORMAT = 'f'
    BYTES_BINARY_FORMAT = _struct.calcsize(BINARY_FORMAT)
    BYTES_ASCII_FORMAT = _ASCII_BYTES_FROM_TYPE['float']
    ASCII_FORMAT = _ASCII_FORMAT_FROM_TYPE['float']
    N_COLUMN = 5

    def __init__(self, x=None):
        self.from_(x)

    @property
    def data(self):
        return getattr(self, '_{}_from_internal'.format(self.meta.iftype))(self._data)

    @data.setter
    def data(self, xs):
        self._data = getattr(self, '_internal_from_{}'.format(self.meta.iftype))(xs)

    def __str__(self):
        ss = []
        for i, x in enumerate(self._data):
            ss.append(self.ASCII_FORMAT.format(x))
            if i%self.N_COLUMN == self.N_COLUMN - 1:
                ss.append('\n')
        return str(self.meta) + ''.join(ss)

    def __bytes__(self):
        return bytes(self.meta) + _struct.pack('{}{}{}'.format(_BINARY_MODE,
                                                               len(self._data),
                                                               self.BINARY_FORMAT),
                                               *self._data)

    def from_(self, x):
        if x is None:
            self.meta = _Meta()
            self._data = []
        elif isinstance(x, (bytes, bytearray)):
            self.from_bytes(x)
        elif isinstance(x, str):
            self.from_str(x)
        elif isinstance(x, dict):
            self.from_dict(x)
        else:
            raise Error('no method to convert {}'.format(type(x)))
        return self

    def from_bytes(self, b):
        self.meta = _Meta().from_bytes(b[:_Meta.BYTES_BINARY_FORMAT])
        bytes_data = len(b) - _Meta.BYTES_BINARY_FORMAT
        assert bytes_data%self.BYTES_BINARY_FORMAT == 0
        self._data = _struct.unpack('{}{}{}'.format(_BINARY_MODE,
                                                    bytes_data//self.BYTES_BINARY_FORMAT,
                                                    self.BINARY_FORMAT),
                                    b[_Meta.BYTES_BINARY_FORMAT:])
        return self

    def from_str(self, s):
        b = s.encode()
        self.meta = _Meta().from_str(b[:_Meta.BYTES_ASCII_FORMAT].decode())
        clean_data = b[_Meta.BYTES_ASCII_FORMAT:].translate(None, b'\n')
        bytes_clean_data = len(clean_data)
        assert pp(bytes_clean_data)%pp(self.BYTES_ASCII_FORMAT) == 0
        self._data = _FLOAT([clean_data[i*self.BYTES_ASCII_FORMAT:(i+1)*self.BYTES_ASCII_FORMAT]
                            for i
                            in range(bytes_clean_data//self.BYTES_ASCII_FORMAT)])
        return self

    def from_dict(self, d):
        self.meta.from_dict(d['meta'])
        self.data = d['data']
        return self

    def to_dict(self):
        return {'meta': self.meta.to_dict(),
                'data': list(self._data)}

    @staticmethod
    def _itime_from_internal(xs):
        """
        [y1, y2, ...] -> [y1, y2, ...]
        """
        return xs

    @staticmethod
    def _internal_from_itime(xs):
        return _assert_floats(xs)

    @staticmethod
    def _ixy_from_internal(xs):
        """
        [y1, y2, ..., x1, x2, ...] -> [[x1, y1], [x2, y2], ...]
        """
        n_xs = len(xs)
        assert n_xs%2 == 0
        return _pylab.transpose(_pylab.reshape(xs, (2, n_xs//2))[::-1])

    @staticmethod
    def _internal_from_ixy(xys):
        n_row, n_column = _pylab.shape(xys)
        assert n_column == 2
        return _pylab.reshape(_pylab.transpose(_assert_floats(xys))[::-1],
                              (n_row*n_column,))

    @staticmethod
    def _iamph_from_internal(xs):
        """
        [r1, r2, ..., θ1, θ2, ...] -> [complex(r1*cos(θ1), r1*sin(θ1)), complex(r2*cos(θ2), r2*sin(θ2)), ...]
        """
        n_xs = len(xs)
        assert n_xs%2 == 0
        rs = xs[:n_xs//2]
        ts = xs[n_xs//2:]
        return rs*_pylab.exp(1j*ts)

    @staticmethod
    def _internal_from_iamph(cs):
        rs = _pylab.absolute(cs)
        ts = _pylab.angle(cs)
        return _assert_floats(_pylab.concatenate((rs, ts)))

    @staticmethod
    def _irlim_from_internal(xs):
        """
        [r1, r2, ..., i1, i2, ...] -> [complex(r1, i1), complex(r2, i2), ...]
        """
        n_xs = len(xs)
        assert(n_xs%2 == 0)
        return xs[:n_xs//2] + 1j*xs[n_xs//2:]

    @staticmethod
    def _internal_from_irlim(cs):
        return _assert_floats(_pylab.concatenate((_pylab.real(cs), _pylab.imag(cs))))


if __name__ == '__main__':
    import unittest

    class Tester(unittest.TestCase):

        def setUp(self):
            self.h = _Meta()

        def test_parse_str():
            s = """\
       1.000000      -2.000000       4.000000      -12345.00      -12345.00
       0.000000       6.000000      -12345.00      -12345.00      -12345.00
      -12345.00      -12345.00      -12345.00      -12345.00      -12345.00
      -12345.00      -12345.00      -12345.00      -12345.00      -12345.00
      -12345.00      -12345.00      -12345.00      -12345.00      -12345.00
      -12345.00      -12345.00      -12345.00      -12345.00      -12345.00
      -12345.00      -12345.00      -12345.00      -12345.00      -12345.00
      -12345.00      -12345.00      -12345.00      -12345.00      -12345.00
      -12345.00      -12345.00      -12345.00      -12345.00      -12345.00
      -12345.00      -12345.00      -12345.00      -12345.00      -12345.00
      -12345.00      -12345.00      -12345.00      -12345.00      -12345.00
      -12345.00       1.000000      -12345.00      -12345.00      -12345.00
      -12345.00      -12345.00      -12345.00      -12345.00      -12345.00
      -12345.00      -12345.00      -12345.00      -12345.00      -12345.00
      2013         1         0        59        59
         1         6    -12345    -12345         7
    -12345    -12345    -12345    -12345    -12345
         1    -12345    -12345    -12345    -12345
    -12345    -12345    -12345    -12345    -12345
    -12345    -12345    -12345    -12345    -12345
    -12345    -12345    -12345    -12345    -12345
         1         0         1         1         0
 筑波 testing_waveform
-12345  -12345  -12345  
-12345  -12345  -12345  
-12345  -12345  -12345  
-12345  -12345  -12345  
-12345  -12345  -12345  
-12345  -12345  -12345  
-12345  -12345  -12345  
       4.000000       3.000000       2.000000       1.000000       0.000000
      -1.000000      -2.000000
"""
            sac_s = Sac(s)
            self.assertAlmostEqual(sac_s.meta.delta, 1)
            self.assertAlmostEqual(sac_s.meta.depmin, -2)
            self.assertAlmostEqual(sac_s.meta.depmax, 4)
            self.assertTrue(sac_s.meta.scale is None)
            self.assertTrue(sac_s.meta.odelta is None)
            self.assertAlmostEqual(sac_s.meta.b, 0)
            self.assertAlmostEqual(sac_s.meta.e, 6)
            self.assertTrue(sac_s.meta.o is None)
            self.assertTrue(sac_s.meta.a is None)
            self.assertTrue(sac_s.meta.fmt is None)
            self.assertTrue(sac_s.meta.t0 is None)
            self.assertTrue(sac_s.meta.t1 is None)
            self.assertTrue(sac_s.meta.t2 is None)
            self.assertTrue(sac_s.meta.t3 is None)
            self.assertTrue(sac_s.meta.t4 is None)
            self.assertTrue(sac_s.meta.t5 is None)
            self.assertTrue(sac_s.meta.t6 is None)
            self.assertTrue(sac_s.meta.t7 is None)
            self.assertTrue(sac_s.meta.t8 is None)
            self.assertTrue(sac_s.meta.t9 is None)
            self.assertTrue(sac_s.meta.f is None)
            self.assertTrue(sac_s.meta.resp is None )
            self.assertTrue(sac_s.meta.resp1 is None)
            self.assertTrue(sac_s.meta.resp2 is None)
            self.assertTrue(sac_s.meta.resp3 is None)
            self.assertTrue(sac_s.meta.resp4 is None)
            self.assertTrue(sac_s.meta.resp5 is None)
            self.assertTrue(sac_s.meta.resp6 is None)
            self.assertTrue(sac_s.meta.resp7 is None)
            self.assertTrue(sac_s.meta.resp8 is None)
            self.assertTrue(sac_s.meta.resp9 is None)
            self.assertTrue(sac_s.meta.stla is None)
            self.assertTrue(sac_s.meta.stlo is None)
            self.assertTrue(sac_s.meta.stel is None)
            self.assertTrue(sac_s.meta.stdp is None)
            self.assertTrue(sac_s.meta.evla is None)
            self.assertTrue(sac_s.meta.evlo is None)
            self.assertTrue(sac_s.meta.evel is None)
            self.assertTrue(sac_s.meta.evdp is None)
            self.assertTrue(sac_s.meta.mag is None)
            self.assertTrue(sac_s.meta.user0 is None)
            self.assertTrue(sac_s.meta.user1 is None)
            self.assertTrue(sac_s.meta.user2 is None)
            self.assertTrue(sac_s.meta.user3 is None)
            self.assertTrue(sac_s.meta.user4 is None)
            self.assertTrue(sac_s.meta.user5 is None)
            self.assertTrue(sac_s.meta.user6 is None)
            self.assertTrue(sac_s.meta.user7 is None)
            self.assertTrue(sac_s.meta.user8 is None)
            self.assertTrue(sac_s.meta.user9 is None)
            self.assertTrue(sac_s.meta.dist is None)
            self.assertTrue(sac_s.meta.az is None)
            self.assertTrue(sac_s.meta.baz is None)
            self.assertTrue(sac_s.meta.gcarc is None)
            self.assertTrue(sac_s.meta.sb is None)
            self.assertTrue(sac_s.meta.sdelta is None)
            self.assertAlmostEqual(sac_s.meta.depmen, 1)
            self.assertTrue(sac_s.meta.cmpaz is None)
            self.assertTrue(sac_s.meta.cmpinc is None)
            self.assertTrue(sac_s.meta.xminimum is None)
            self.assertTrue(sac_s.meta.xmaximum is None)
            self.assertTrue(sac_s.meta.yminimum is None)
            self.assertTrue(sac_s.meta.ymaximum is None)
            self.assertTrue(sac_s.meta.fhdr64 is None)
            self.assertTrue(sac_s.meta.fhdr65 is None)
            self.assertTrue(sac_s.meta.fhdr66 is None)
            self.assertTrue(sac_s.meta.fhdr67 is None)
            self.assertTrue(sac_s.meta.fhdr68 is None)
            self.assertTrue(sac_s.meta.fhdr69 is None)
            self.assertTrue(sac_s.meta.fhdr70 is None)
            self.assertEqual(sac_s.meta.nzyear, 2013)
            self.assertEqual(sac_s.meta.nzjday, 1)
            self.assertEqual(sac_s.meta.nzhour, 0)
            self.assertEqual(sac_s.meta.nzmin, 59)
            self.assertEqual(sac_s.meta.nzsec, 59)
            self.assertEqual(sac_s.meta.nzmsec, 1)
            self.assertEqual(sac_s.meta.nvhdr, 6)
            self.assertTrue(sac_s.meta.norid is None)
            self.assertTrue(sac_s.meta.nevid is None)
            self.assertEqual(sac_s.meta.npts, 7)
            self.assertTrue(sac_s.meta.nspts is None)
            self.assertTrue(sac_s.meta.nwfid is None)
            self.assertTrue(sac_s.meta.nxsize is None)
            self.assertTrue(sac_s.meta.nysize is None)
            self.assertTrue(sac_s.meta.nhdr56 is None)
            self.assertAlmostEqual(sac_s.meta.iftype, 'itime')
            self.assertTrue(sac_s.meta.idep is None)
            self.assertTrue(sac_s.meta.iztype is None)
            self.assertTrue(sac_s.meta.ihdr4 is None)
            self.assertTrue(sac_s.meta.iinst is None)
            self.assertTrue(sac_s.meta.istreg is None)
            self.assertTrue(sac_s.meta.ievreg is None)
            self.assertTrue(sac_s.meta.ievtyp is None)
            self.assertTrue(sac_s.meta.iqual is None)
            self.assertTrue(sac_s.meta.isynth is None)
            self.assertTrue(sac_s.meta.imagtyp is None)
            self.assertTrue(sac_s.meta.imagsrc is None)
            self.assertTrue(sac_s.meta.ihdr13 is None)
            self.assertTrue(sac_s.meta.ihdr14 is None)
            self.assertTrue(sac_s.meta.ihdr15 is None)
            self.assertTrue(sac_s.meta.ihdr16 is None)
            self.assertTrue(sac_s.meta.ihdr17 is None)
            self.assertTrue(sac_s.meta.ihdr18 is None)
            self.assertTrue(sac_s.meta.ihdr19 is None)
            self.assertTrue(sac_s.meta.ihdr20 is None)
            self.assertTrue(sac_s.meta.leven)
            self.assertFalse(sac_s.meta.lpspol)
            self.assertTrue(sac_s.meta.lovrok)
            self.assertTrue(sac_s.meta.lcalda)
            self.assertFalse(sac_s.meta.lhdr5)
            self.assertEqual(sac_s.meta.kstnm, ' 筑波')
            self.assertEqual(sac_s.meta.kevnm, 'testing_waveform')
            self.assertTrue(sac_s.meta.khole is None)
            self.assertTrue(sac_s.meta.ko is None)
            self.assertTrue(sac_s.meta.ka is None)
            self.assertTrue(sac_s.meta.kt0 is None)
            self.assertTrue(sac_s.meta.kt1 is None)
            self.assertTrue(sac_s.meta.kt2 is None)
            self.assertTrue(sac_s.meta.kt3 is None)
            self.assertTrue(sac_s.meta.kt4 is None)
            self.assertTrue(sac_s.meta.kt5 is None)
            self.assertTrue(sac_s.meta.kt6 is None)
            self.assertTrue(sac_s.meta.kt7 is None)
            self.assertTrue(sac_s.meta.kt8 is None)
            self.assertTrue(sac_s.meta.kt9 is None)
            self.assertTrue(sac_s.meta.kf is None)
            self.assertTrue(sac_s.meta.kuser0 is None)
            self.assertTrue(sac_s.meta.kuser1 is None)
            self.assertTrue(sac_s.meta.kuser2 is None)
            self.assertTrue(sac_s.meta.kcmpnm is None)
            self.assertTrue(sac_s.meta.knetwk is None)
            self.assertTrue(sac_s.meta.kdatrd is None)
            self.assertTrue(sac_s.meta.kinst is None)
            self.assertEqual(len(sac_s.data), 7)
            for parsed, orig in zip(sac_s.data, [4, 3, 2, 1, 0, -1, -2]):
                self.assertAlmostEqual(parsed, orig)
            self.assertEqual(str(sac_s), s)

        def test_parse_bytes():
            b = b'\x00\x00\x80?\x00\x00\x00\xc0\x00\x00\x80@\x00\xe4@\xc6\x00\xe4@\xc6\x00\x00\x00\x00\x00\x00\xc0@\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\x00\x80?\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\x06\x00\x00\x00\xc7\xcf\xff\xff\xc7\xcf\xff\xff\x07\x00\x00\x00\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\x01\x00\x00\x00\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00-12345  testing_waveform-12345  -12345  -12345  -12345  -12345  -12345  -12345  -12345  -12345  -12345  -12345  -12345  -12345  -12345  -12345  -12345  -12345  -12345  -12345  -12345  -12345  \x00\x00\x80@\x00\x00@@\x00\x00\x00@\x00\x00\x80?\x00\x00\x00\x00\x00\x00\x80\xbf\x00\x00\x00\xc0'
            sac_b = Sac(b)
            self.assertAlmostEqual(sac_b.meta.delta, 1)
            self.assertAlmostEqual(sac_b.meta.depmin, -2)
            self.assertAlmostEqual(sac_b.meta.depmax, 4)
            self.assertTrue(sac_b.meta.scale is None)
            self.assertTrue(sac_b.meta.odelta is None)
            self.assertAlmostEqual(sac_b.meta.b, 0)
            self.assertAlmostEqual(sac_b.meta.e, 6)
            self.assertTrue(sac_b.meta.o is None)
            self.assertTrue(sac_b.meta.a is None)
            self.assertTrue(sac_b.meta.fmt is None)
            self.assertTrue(sac_b.meta.t0 is None)
            self.assertTrue(sac_b.meta.t1 is None)
            self.assertTrue(sac_b.meta.t2 is None)
            self.assertTrue(sac_b.meta.t3 is None)
            self.assertTrue(sac_b.meta.t4 is None)
            self.assertTrue(sac_b.meta.t5 is None)
            self.assertTrue(sac_b.meta.t6 is None)
            self.assertTrue(sac_b.meta.t7 is None)
            self.assertTrue(sac_b.meta.t8 is None)
            self.assertTrue(sac_b.meta.t9 is None)
            self.assertTrue(sac_b.meta.f is None)
            self.assertTrue(sac_b.meta.resp is None )
            self.assertTrue(sac_b.meta.resp1 is None)
            self.assertTrue(sac_b.meta.resp2 is None)
            self.assertTrue(sac_b.meta.resp3 is None)
            self.assertTrue(sac_b.meta.resp4 is None)
            self.assertTrue(sac_b.meta.resp5 is None)
            self.assertTrue(sac_b.meta.resp6 is None)
            self.assertTrue(sac_b.meta.resp7 is None)
            self.assertTrue(sac_b.meta.resp8 is None)
            self.assertTrue(sac_b.meta.resp9 is None)
            self.assertTrue(sac_b.meta.stla is None)
            self.assertTrue(sac_b.meta.stlo is None)
            self.assertTrue(sac_b.meta.stel is None)
            self.assertTrue(sac_b.meta.stdp is None)
            self.assertTrue(sac_b.meta.evla is None)
            self.assertTrue(sac_b.meta.evlo is None)
            self.assertTrue(sac_b.meta.evel is None)
            self.assertTrue(sac_b.meta.evdp is None)
            self.assertTrue(sac_b.meta.mag is None)
            self.assertTrue(sac_b.meta.user0 is None)
            self.assertTrue(sac_b.meta.user1 is None)
            self.assertTrue(sac_b.meta.user2 is None)
            self.assertTrue(sac_b.meta.user3 is None)
            self.assertTrue(sac_b.meta.user4 is None)
            self.assertTrue(sac_b.meta.user5 is None)
            self.assertTrue(sac_b.meta.user6 is None)
            self.assertTrue(sac_b.meta.user7 is None)
            self.assertTrue(sac_b.meta.user8 is None)
            self.assertTrue(sac_b.meta.user9 is None)
            self.assertTrue(sac_b.meta.dist is None)
            self.assertTrue(sac_b.meta.az is None)
            self.assertTrue(sac_b.meta.baz is None)
            self.assertTrue(sac_b.meta.gcarc is None)
            self.assertTrue(sac_b.meta.sb is None)
            self.assertTrue(sac_b.meta.sdelta is None)
            self.assertAlmostEqual(sac_b.meta.depmen, 1)
            self.assertTrue(sac_b.meta.cmpaz is None)
            self.assertTrue(sac_b.meta.cmpinc is None)
            self.assertTrue(sac_b.meta.xminimum is None)
            self.assertTrue(sac_b.meta.xmaximum is None)
            self.assertTrue(sac_b.meta.yminimum is None)
            self.assertTrue(sac_b.meta.ymaximum is None)
            self.assertTrue(sac_b.meta.fhdr64 is None)
            self.assertTrue(sac_b.meta.fhdr65 is None)
            self.assertTrue(sac_b.meta.fhdr66 is None)
            self.assertTrue(sac_b.meta.fhdr67 is None)
            self.assertTrue(sac_b.meta.fhdr68 is None)
            self.assertTrue(sac_b.meta.fhdr69 is None)
            self.assertTrue(sac_b.meta.fhdr70 is None)
            self.assertEqual(sac_b.meta.nzyear, 2013)
            self.assertEqual(sac_b.meta.nzjday, 1)
            self.assertEqual(sac_b.meta.nzhour, 0)
            self.assertEqual(sac_b.meta.nzmin, 59)
            self.assertEqual(sac_b.meta.nzsec, 59)
            self.assertEqual(sac_b.meta.nzmsec, 1)
            self.assertEqual(sac_b.meta.nvhdr, 6)
            self.assertTrue(sac_b.meta.norid is None)
            self.assertTrue(sac_b.meta.nevid is None)
            self.assertEqual(sac_b.meta.npts, 7)
            self.assertTrue(sac_b.meta.nspts is None)
            self.assertTrue(sac_b.meta.nwfid is None)
            self.assertTrue(sac_b.meta.nxsize is None)
            self.assertTrue(sac_b.meta.nysize is None)
            self.assertTrue(sac_b.meta.nhdr56 is None)
            self.assertAlmostEqual(sac_b.meta.iftype, 'itime')
            self.assertTrue(sac_b.meta.idep is None)
            self.assertTrue(sac_b.meta.iztype is None)
            self.assertTrue(sac_b.meta.ihdr4 is None)
            self.assertTrue(sac_b.meta.iinst is None)
            self.assertTrue(sac_b.meta.istreg is None)
            self.assertTrue(sac_b.meta.ievreg is None)
            self.assertTrue(sac_b.meta.ievtyp is None)
            self.assertTrue(sac_b.meta.iqual is None)
            self.assertTrue(sac_b.meta.isynth is None)
            self.assertTrue(sac_b.meta.imagtyp is None)
            self.assertTrue(sac_b.meta.imagsrc is None)
            self.assertTrue(sac_b.meta.ihdr13 is None)
            self.assertTrue(sac_b.meta.ihdr14 is None)
            self.assertTrue(sac_b.meta.ihdr15 is None)
            self.assertTrue(sac_b.meta.ihdr16 is None)
            self.assertTrue(sac_b.meta.ihdr17 is None)
            self.assertTrue(sac_b.meta.ihdr18 is None)
            self.assertTrue(sac_b.meta.ihdr19 is None)
            self.assertTrue(sac_b.meta.ihdr20 is None)
            self.assertTrue(sac_b.meta.leven)
            self.assertFalse(sac_b.meta.lpspol)
            self.assertTrue(sac_b.meta.lovrok)
            self.assertTrue(sac_b.meta.lcalda)
            self.assertFalse(sac_b.meta.lhdr5)
            self.assertEqual(sac_b.meta.kstnm, ' 筑波')
            self.assertEqual(sac_b.meta.kevnm, 'testing_waveform')
            self.assertTrue(sac_b.meta.khole is None)
            self.assertTrue(sac_b.meta.ko is None)
            self.assertTrue(sac_b.meta.ka is None)
            self.assertTrue(sac_b.meta.kt0 is None)
            self.assertTrue(sac_b.meta.kt1 is None)
            self.assertTrue(sac_b.meta.kt2 is None)
            self.assertTrue(sac_b.meta.kt3 is None)
            self.assertTrue(sac_b.meta.kt4 is None)
            self.assertTrue(sac_b.meta.kt5 is None)
            self.assertTrue(sac_b.meta.kt6 is None)
            self.assertTrue(sac_b.meta.kt7 is None)
            self.assertTrue(sac_b.meta.kt8 is None)
            self.assertTrue(sac_b.meta.kt9 is None)
            self.assertTrue(sac_b.meta.kf is None)
            self.assertTrue(sac_b.meta.kuser0 is None)
            self.assertTrue(sac_b.meta.kuser1 is None)
            self.assertTrue(sac_b.meta.kuser2 is None)
            self.assertTrue(sac_b.meta.kcmpnm is None)
            self.assertTrue(sac_b.meta.knetwk is None)
            self.assertTrue(sac_b.meta.kdatrd is None)
            self.assertTrue(sac_b.meta.kinst is None)
            self.assertEqual(len(sac_b.data), 7)
            for parsed, orig in zip(sac_b.data, [4, 3, 2, 1, 0, -1, -2]):
                self.assertAlmostEqual(parsed, orig)
            self.assertEqual(bytes(sac_b), b)

        def test_parse_dict():
            d = {
                'meta': {
                    'delta': 1,
                    'depmin': -2,
                    'depmax': 4,
                    'scale': None,
                    'odelta': None,
                    'b': 0,
                    'e': 6,
                    'o': None,
                    'a': None,
                    'fmt': None,
                    't0': None,
                    't1': None,
                    't2': None,
                    't3': None,
                    't4': None,
                    't5': None,
                    't6': None,
                    't7': None,
                    't8': None,
                    't9': None,
                    'f': None,
                    'resp0': None,
                    'resp1': None,
                    'resp2': None,
                    'resp3': None,
                    'resp4': None,
                    'resp5': None,
                    'resp6': None,
                    'resp7': None,
                    'resp8': None,
                    'resp9': None,
                    'stla': None,
                    'stlo': None,
                    'stel': None,
                    'stdp': None,
                    'evla': None,
                    'evlo': None,
                    'evel': None,
                    'evdp': None,
                    'mag': None,
                    'user0': None,
                    'user1': None,
                    'user2': None,
                    'user3': None,
                    'user4': None,
                    'user5': None,
                    'user6': None,
                    'user7': None,
                    'user8': None,
                    'user9': None,
                    'dist': None,
                    'az': None,
                    'baz': None,
                    'gcarc': None,
                    'sb': None,
                    'sdelta': None,
                    'depmen': 1,
                    'cmpaz': None,
                    'cmpinc': None,
                    'xminimum': None,
                    'xmaximum': None,
                    'yminimum': None,
                    'ymaximum': None,
                    'fhdr64': None,
                    'fhdr65': None,
                    'fhdr66': None,
                    'fhdr67': None,
                    'fhdr68': None,
                    'fhdr69': None,
                    'fhdr70': None,
                    'nzyear': 2013,
                    'nzjday': 1,
                    'nzhour': 0,
                    'nzmin': 59,
                    'nzsec': 59,
                    'nzmsec': 1,
                    'nvhdr': 6,
                    'norid': None,
                    'nevid': None,
                    'npts': 6,
                    'nspts': None,
                    'nwfid': None,
                    'nxsize': None,
                    'nysize': None,
                    'nhdr56': None,
                    'iftype': 'itime',
                    'idep': None,
                    'iztype': None,
                    'ihdr4': None,
                    'iinst': None,
                    'istreg': None,
                    'ievreg': None,
                    'ievtyp': None,
                    'iqual': None,
                    'isynth': None,
                    'imagtyp': None,
                    'imagsrc': None,
                    'ihdr13': None,
                    'ihdr14': None,
                    'ihdr15': None,
                    'ihdr16': None,
                    'ihdr17': None,
                    'ihdr18': None,
                    'ihdr19': None,
                    'ihdr20': None,
                    'leven': True,
                    'lpspol': False,
                    'lovrok': True,
                    'lcalda': True,
                    'lhdr5': False,
                    'kstnm': ' 筑波',
                    'kevnm': 'testing_waveform',
                    'khole': None,
                    'ko': None,
                    'ka': None,
                    'kt0': None,
                    'kt1': None,
                    'kt2': None,
                    'kt3': None,
                    'kt4': None,
                    'kt5': None,
                    'kt6': None,
                    'kt7': None,
                    'kt8': None,
                    'kt9': None,
                    'kf': None,
                    'kuser0': None,
                    'kuser1': None,
                    'kuser2': None,
                    'kcmpnm': None,
                    'knetwk': None,
                    'kdatrd': None,
                    'kinst': None,
                },
                'data': [4, 3, 2, 1, 0, -1, -2]
            }
            sac_d = Sac(d)
            self.assertAlmostEqual(sac_d.meta.delta, 1)
            self.assertAlmostEqual(sac_d.meta.depmin, -2)
            self.assertAlmostEqual(sac_d.meta.depmax, 4)
            self.assertTrue(sac_d.meta.scale is None)
            self.assertTrue(sac_d.meta.odelta is None)
            self.assertAlmostEqual(sac_d.meta.b, 0)
            self.assertAlmostEqual(sac_d.meta.e, 6)
            self.assertTrue(sac_d.meta.o is None)
            self.assertTrue(sac_d.meta.a is None)
            self.assertTrue(sac_d.meta.fmt is None)
            self.assertTrue(sac_d.meta.t0 is None)
            self.assertTrue(sac_d.meta.t1 is None)
            self.assertTrue(sac_d.meta.t2 is None)
            self.assertTrue(sac_d.meta.t3 is None)
            self.assertTrue(sac_d.meta.t4 is None)
            self.assertTrue(sac_d.meta.t5 is None)
            self.assertTrue(sac_d.meta.t6 is None)
            self.assertTrue(sac_d.meta.t7 is None)
            self.assertTrue(sac_d.meta.t8 is None)
            self.assertTrue(sac_d.meta.t9 is None)
            self.assertTrue(sac_d.meta.f is None)
            self.assertTrue(sac_d.meta.resp is None )
            self.assertTrue(sac_d.meta.resp1 is None)
            self.assertTrue(sac_d.meta.resp2 is None)
            self.assertTrue(sac_d.meta.resp3 is None)
            self.assertTrue(sac_d.meta.resp4 is None)
            self.assertTrue(sac_d.meta.resp5 is None)
            self.assertTrue(sac_d.meta.resp6 is None)
            self.assertTrue(sac_d.meta.resp7 is None)
            self.assertTrue(sac_d.meta.resp8 is None)
            self.assertTrue(sac_d.meta.resp9 is None)
            self.assertTrue(sac_d.meta.stla is None)
            self.assertTrue(sac_d.meta.stlo is None)
            self.assertTrue(sac_d.meta.stel is None)
            self.assertTrue(sac_d.meta.stdp is None)
            self.assertTrue(sac_d.meta.evla is None)
            self.assertTrue(sac_d.meta.evlo is None)
            self.assertTrue(sac_d.meta.evel is None)
            self.assertTrue(sac_d.meta.evdp is None)
            self.assertTrue(sac_d.meta.mag is None)
            self.assertTrue(sac_d.meta.user0 is None)
            self.assertTrue(sac_d.meta.user1 is None)
            self.assertTrue(sac_d.meta.user2 is None)
            self.assertTrue(sac_d.meta.user3 is None)
            self.assertTrue(sac_d.meta.user4 is None)
            self.assertTrue(sac_d.meta.user5 is None)
            self.assertTrue(sac_d.meta.user6 is None)
            self.assertTrue(sac_d.meta.user7 is None)
            self.assertTrue(sac_d.meta.user8 is None)
            self.assertTrue(sac_d.meta.user9 is None)
            self.assertTrue(sac_d.meta.dist is None)
            self.assertTrue(sac_d.meta.az is None)
            self.assertTrue(sac_d.meta.baz is None)
            self.assertTrue(sac_d.meta.gcarc is None)
            self.assertTrue(sac_d.meta.sb is None)
            self.assertTrue(sac_d.meta.sdelta is None)
            self.assertAlmostEqual(sac_d.meta.depmen, 1)
            self.assertTrue(sac_d.meta.cmpaz is None)
            self.assertTrue(sac_d.meta.cmpinc is None)
            self.assertTrue(sac_d.meta.xminimum is None)
            self.assertTrue(sac_d.meta.xmaximum is None)
            self.assertTrue(sac_d.meta.yminimum is None)
            self.assertTrue(sac_d.meta.ymaximum is None)
            self.assertTrue(sac_d.meta.fhdr64 is None)
            self.assertTrue(sac_d.meta.fhdr65 is None)
            self.assertTrue(sac_d.meta.fhdr66 is None)
            self.assertTrue(sac_d.meta.fhdr67 is None)
            self.assertTrue(sac_d.meta.fhdr68 is None)
            self.assertTrue(sac_d.meta.fhdr69 is None)
            self.assertTrue(sac_d.meta.fhdr70 is None)
            self.assertEqual(sac_d.meta.nzyear, 2013)
            self.assertEqual(sac_d.meta.nzjday, 1)
            self.assertEqual(sac_d.meta.nzhour, 0)
            self.assertEqual(sac_d.meta.nzmin, 59)
            self.assertEqual(sac_d.meta.nzsec, 59)
            self.assertEqual(sac_d.meta.nzmsec, 1)
            self.assertEqual(sac_d.meta.nvhdr, 6)
            self.assertTrue(sac_d.meta.norid is None)
            self.assertTrue(sac_d.meta.nevid is None)
            self.assertEqual(sac_d.meta.npts, 7)
            self.assertTrue(sac_d.meta.nspts is None)
            self.assertTrue(sac_d.meta.nwfid is None)
            self.assertTrue(sac_d.meta.nxsize is None)
            self.assertTrue(sac_d.meta.nysize is None)
            self.assertTrue(sac_d.meta.nhdr56 is None)
            self.assertAlmostEqual(sac_d.meta.iftype, 'itime')
            self.assertTrue(sac_d.meta.idep is None)
            self.assertTrue(sac_d.meta.iztype is None)
            self.assertTrue(sac_d.meta.ihdr4 is None)
            self.assertTrue(sac_d.meta.iinst is None)
            self.assertTrue(sac_d.meta.istreg is None)
            self.assertTrue(sac_d.meta.ievreg is None)
            self.assertTrue(sac_d.meta.ievtyp is None)
            self.assertTrue(sac_d.meta.iqual is None)
            self.assertTrue(sac_d.meta.isynth is None)
            self.assertTrue(sac_d.meta.imagtyp is None)
            self.assertTrue(sac_d.meta.imagsrc is None)
            self.assertTrue(sac_d.meta.ihdr13 is None)
            self.assertTrue(sac_d.meta.ihdr14 is None)
            self.assertTrue(sac_d.meta.ihdr15 is None)
            self.assertTrue(sac_d.meta.ihdr16 is None)
            self.assertTrue(sac_d.meta.ihdr17 is None)
            self.assertTrue(sac_d.meta.ihdr18 is None)
            self.assertTrue(sac_d.meta.ihdr19 is None)
            self.assertTrue(sac_d.meta.ihdr20 is None)
            self.assertTrue(sac_d.meta.leven)
            self.assertFalse(sac_d.meta.lpspol)
            self.assertTrue(sac_d.meta.lovrok)
            self.assertTrue(sac_d.meta.lcalda)
            self.assertFalse(sac_d.meta.lhdr5)
            self.assertEqual(sac_d.meta.kstnm, ' 筑波')
            self.assertEqual(sac_d.meta.kevnm, 'testing_waveform')
            self.assertTrue(sac_d.meta.khole is None)
            self.assertTrue(sac_d.meta.ko is None)
            self.assertTrue(sac_d.meta.ka is None)
            self.assertTrue(sac_d.meta.kt0 is None)
            self.assertTrue(sac_d.meta.kt1 is None)
            self.assertTrue(sac_d.meta.kt2 is None)
            self.assertTrue(sac_d.meta.kt3 is None)
            self.assertTrue(sac_d.meta.kt4 is None)
            self.assertTrue(sac_d.meta.kt5 is None)
            self.assertTrue(sac_d.meta.kt6 is None)
            self.assertTrue(sac_d.meta.kt7 is None)
            self.assertTrue(sac_d.meta.kt8 is None)
            self.assertTrue(sac_d.meta.kt9 is None)
            self.assertTrue(sac_d.meta.kf is None)
            self.assertTrue(sac_d.meta.kuser0 is None)
            self.assertTrue(sac_d.meta.kuser1 is None)
            self.assertTrue(sac_d.meta.kuser2 is None)
            self.assertTrue(sac_d.meta.kcmpnm is None)
            self.assertTrue(sac_d.meta.knetwk is None)
            self.assertTrue(sac_d.meta.kdatrd is None)
            self.assertTrue(sac_d.meta.kinst is None)
            self.assertEqual(len(sac_d.data), 7)
            for parsed, orig in zip(sac_d.data, [4, 3, 2, 1, 0, -1, -2]):
                self.assertAlmostEqual(parsed, orig)
            d_out = sac_d.to_dict()
            self.assertEqual(len(d_out['data']), 7)
            for parsed, orig in zip(d_out['data'], d['data']):
                self.assertAlmostEqual(parsed, orig)
            for key in d.keys():
                self.assertAlmostEqual(d_out[key], d[key])

        def test_ascii(self):
            self.assertEqual(str(self.h), str(_Meta().from_str(str(self.h))))
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
            self.h.lovrok = None
            self.h.nvhdr = None
            self.assertEqual(str(self.h), s)
            with self.assertRaises(AssertionError):
                self.h.from_str('too short')

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
