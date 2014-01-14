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


class Meta(object):

    INTERNAL_LEN_FROM_TYPE = {
        'short_string': 8,
        'long_string': 16,
    }
    LEN_SHORT_STRING = INTERNAL_LEN_FROM_TYPE['short_string']
    LEN_LONG_STRING = INTERNAL_LEN_FROM_TYPE['long_string']

    ASCII_WIDTH_FROM_TYPE = {
        'logical': 10,
        'integer': 10,
        'float': 15,
        'enum': 10,
        'short_string': LEN_SHORT_STRING,
        'long_string': LEN_LONG_STRING,
    }

    ASCII_FORMAT_FROM_TYPE = {
        'logical': '{{:>{}d}}'.format(ASCII_WIDTH_FROM_TYPE['logical']),
        'integer': '{{:>{}d}}'.format(ASCII_WIDTH_FROM_TYPE['integer']),
        'float': '{{:>#{}.7g}}'.format(ASCII_WIDTH_FROM_TYPE['float']),
        'enum': '{{:>{}d}}'.format(ASCII_WIDTH_FROM_TYPE['enum']),
        'short_string': '{{:<{}s}}'.format(LEN_SHORT_STRING),
        'long_string': '{{:<{}s}}'.format(LEN_LONG_STRING),
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
        'short_string': '{}c'.format(LEN_SHORT_STRING),
        'long_string': '{}c'.format(LEN_LONG_STRING),
    }

    UNDEFINED_FROM_TYPE = {
        'logical': -12345,
        'integer': -12345,
        'float': -12345.0,
        'enum': -12345,
        'short_string': ASCII_FORMAT_FROM_TYPE['short_string'].format('-12345'),
        'long_string': ASCII_FORMAT_FROM_TYPE['long_string'].format('-12345'),
    }
    UNDEFINED_LOGICAL = UNDEFINED_FROM_TYPE['logical']
    UNDEFINED_INTEGER = UNDEFINED_FROM_TYPE['integer']
    UNDEFINED_FLOAT = UNDEFINED_FROM_TYPE['float']
    UNDEFINED_ENUM = UNDEFINED_FROM_TYPE['enum']
    UNDEFINED_SHORT_STRING = UNDEFINED_FROM_TYPE['short_string']
    UNDEFINED_LONG_STRING = UNDEFINED_FROM_TYPE['long_string']

    FIELDS = (
        {'name': 'delta', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'depmin', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'depmax', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'scale', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'odelta', 'type': 'float', 'eol': True, 'default': None},
        {'name': 'b', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'e', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'o', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'a', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'internal1', 'type': 'float', 'eol': True, 'default': None}, # fmt
        {'name': 't0', 'type': 'float', 'eol': False, 'default': None},
        {'name': 't1', 'type': 'float', 'eol': False, 'default': None},
        {'name': 't2', 'type': 'float', 'eol': False, 'default': None},
        {'name': 't3', 'type': 'float', 'eol': False, 'default': None},
        {'name': 't4', 'type': 'float', 'eol': True, 'default': None},
        {'name': 't5', 'type': 'float', 'eol': False, 'default': None},
        {'name': 't6', 'type': 'float', 'eol': False, 'default': None},
        {'name': 't7', 'type': 'float', 'eol': False, 'default': None},
        {'name': 't8', 'type': 'float', 'eol': False, 'default': None},
        {'name': 't9', 'type': 'float', 'eol': True, 'default': None},
        {'name': 'f', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'resp0', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'resp1', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'resp2', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'resp3', 'type': 'float', 'eol': True, 'default': None},
        {'name': 'resp4', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'resp5', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'resp6', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'resp7', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'resp8', 'type': 'float', 'eol': True, 'default': None},
        {'name': 'resp9', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'stla', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'stlo', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'stel', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'stdp', 'type': 'float', 'eol': True, 'default': None},
        {'name': 'evla', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'evlo', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'evel', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'evdp', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'mag', 'type': 'float', 'eol': True, 'default': None},
        {'name': 'user0', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'user1', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'user2', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'user3', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'user4', 'type': 'float', 'eol': True, 'default': None},
        {'name': 'user5', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'user6', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'user7', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'user8', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'user9', 'type': 'float', 'eol': True, 'default': None},
        {'name': 'dist', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'az', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'baz', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'gcarc', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'internal2', 'type': 'float', 'eol': True, 'default': None}, # sb
        {'name': 'internal3', 'type': 'float', 'eol': False, 'default': None}, # sdelta
        {'name': 'depmen', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'cmpaz', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'cmpinc', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'xminimum', 'type': 'float', 'eol': True, 'default': None},
        {'name': 'xmaximum', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'yminimum', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'ymaximum', 'type': 'float', 'eol': False, 'default': None},
        {'name': 'adjtm', 'type': 'float', 'eol': False, 'default': None}, # fhdr64
        {'name': 'unused1', 'type': 'float', 'eol': True, 'default': None}, # fhdr65
        {'name': 'unused2', 'type': 'float', 'eol': False, 'default': None}, # fhdr66
        {'name': 'unused3', 'type': 'float', 'eol': False, 'default': None}, # fhdr67
        {'name': 'unused4', 'type': 'float', 'eol': False, 'default': None}, # fhdr68
        {'name': 'unused5', 'type': 'float', 'eol': False, 'default': None}, # fhdr69
        {'name': 'unused6', 'type': 'float', 'eol': True, 'default': None}, # fhdr70
        {'name': 'nzyear', 'type': 'integer', 'eol': False, 'default': None},
        {'name': 'nzjday', 'type': 'integer', 'eol': False, 'default': None},
        {'name': 'nzhour', 'type': 'integer', 'eol': False, 'default': None},
        {'name': 'nzmin', 'type': 'integer', 'eol': False, 'default': None},
        {'name': 'nzsec', 'type': 'integer', 'eol': True, 'default': None},
        {'name': 'nzmsec', 'type': 'integer', 'eol': False, 'default': None},
        {'name': 'nvhdr', 'type': 'integer', 'eol': False, 'default': None},
        {'name': 'norid', 'type': 'integer', 'eol': False, 'default': None},
        {'name': 'nevid', 'type': 'integer', 'eol': False, 'default': None},
        {'name': 'npts', 'type': 'integer', 'eol': True, 'default': None},
        {'name': 'nspts', 'type': 'integer', 'eol': False, 'default': None},
        {'name': 'nwfid', 'type': 'integer', 'eol': False, 'default': None},
        {'name': 'nxsize', 'type': 'integer', 'eol': False, 'default': None},
        {'name': 'nysize', 'type': 'integer', 'eol': False, 'default': None},
        {'name': 'unused7', 'type': 'integer', 'eol': True, 'default': None}, # nhdr56
        {'name': 'iftype', 'type': 'enum', 'eol': False, 'default':None},
        {'name': 'idep', 'type': 'enum', 'eol': False, 'default':None},
        {'name': 'iztype', 'type': 'enum', 'eol': False, 'default':None},
        {'name': 'unused8', 'type': 'enum', 'eol': False, 'default':None}, # ihdr4
        {'name': 'iinst', 'type': 'enum', 'eol': True, 'default':None},
        {'name': 'istreg', 'type': 'enum', 'eol': False, 'default':None},
        {'name': 'ievreg', 'type': 'enum', 'eol': False, 'default':None},
        {'name': 'ievtyp', 'type': 'enum', 'eol': False, 'default':None},
        {'name': 'iqual', 'type': 'enum', 'eol': False, 'default':None},
        {'name': 'isynth', 'type': 'enum', 'eol': True, 'default':None},
        {'name': 'imagtyp', 'type': 'enum', 'eol': False, 'default':None},
        {'name': 'imagsrc', 'type': 'enum', 'eol': False, 'default':None},
        {'name': 'unused9', 'type': 'enum', 'eol': False, 'default':None}, # ihdr13
        {'name': 'unused10', 'type': 'enum', 'eol': False, 'default':None}, # ihdr14
        {'name': 'unused11', 'type': 'enum', 'eol': True, 'default':None}, # ihdr15
        {'name': 'unused12', 'type': 'enum', 'eol': False, 'default':None}, # ihdr16
        {'name': 'unused13', 'type': 'enum', 'eol': False, 'default':None}, # ihdr17
        {'name': 'unused14', 'type': 'enum', 'eol': False, 'default':None}, # ihdr18
        {'name': 'unused15', 'type': 'enum', 'eol': False, 'default':None}, # ihdr19
        {'name': 'unused16', 'type': 'enum', 'eol': True, 'default':None}, # ihdr20
        {'name': 'leven', 'type': 'logical', 'eol': False, 'default': None},
        {'name': 'lpspol', 'type': 'logical', 'eol': False, 'default': None},
        {'name': 'lovrok', 'type': 'logical', 'eol': False, 'default': None},
        {'name': 'lcalda', 'type': 'logical', 'eol': False, 'default': None},
        {'name': 'unused17', 'type': 'logical', 'eol': True, 'default': None}, # lhdr5
        {'name': 'kstnm', 'type': 'short_string', 'eol': False, 'default': None},
        {'name': 'kevnm', 'type': 'long_string', 'eol': True, 'default': None},
        {'name': 'khole', 'type': 'short_string', 'eol': False, 'default': None},
        {'name': 'ko', 'type': 'short_string', 'eol': False, 'default': None},
        {'name': 'ka', 'type': 'short_string', 'eol': True, 'default': None},
        {'name': 'kt0', 'type': 'short_string', 'eol': False, 'default': None},
        {'name': 'kt1', 'type': 'short_string', 'eol': False, 'default': None},
        {'name': 'kt2', 'type': 'short_string', 'eol': True, 'default': None},
        {'name': 'kt3', 'type': 'short_string', 'eol': False, 'default': None},
        {'name': 'kt4', 'type': 'short_string', 'eol': False, 'default': None},
        {'name': 'kt5', 'type': 'short_string', 'eol': True, 'default': None},
        {'name': 'kt6', 'type': 'short_string', 'eol': False, 'default': None},
        {'name': 'kt7', 'type': 'short_string', 'eol': False, 'default': None},
        {'name': 'kt8', 'type': 'short_string', 'eol': True, 'default': None},
        {'name': 'kt9', 'type': 'short_string', 'eol': False, 'default': None},
        {'name': 'kf', 'type': 'short_string', 'eol': False, 'default': None},
        {'name': 'kuser0', 'type': 'short_string', 'eol': True, 'default': None},
        {'name': 'kuser1', 'type': 'short_string', 'eol': False, 'default': None},
        {'name': 'kuser2', 'type': 'short_string', 'eol': False, 'default': None},
        {'name': 'kcmpnm', 'type': 'short_string', 'eol': True, 'default': None},
        {'name': 'knetwk', 'type': 'short_string', 'eol': False, 'default': None},
        {'name': 'kdatrd', 'type': 'short_string', 'eol': False, 'default': None},
        {'name': 'kinst', 'type': 'short_string', 'eol': True, 'default': None},
    )

    NAMES = tuple(f['name'] for f in FIELDS)
    TYPE_FROM_NAMES = {f['name']: f['type'] for f in FIELDS}

    ASCII_FORMAT = []
    BINARY_FORMAT = []
    for field in FIELDS:
        t = field['type']
        ascii_format = ASCII_FORMAT_FROM_TYPE[t]
        if field['eol']:
            ascii_format += '\n'
        ASCII_FORMAT.append(ascii_format)
        BINARY_FORMAT.append(BINARY_FORMAT_FROM_TYPE[t])
    ASCII_FORMAT = ''.join(ASCII_FORMAT)
    BINARY_FORMAT = ''.join(BINARY_FORMAT)

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

    def __init__(self):
        for field in self.FIELDS:
            setattr(self, field['name'], field['default'])

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
        return self.ASCII_FORMAT.format(*[getattr(self, self._internal_name(field['name'])) for field in self.FIELDS])

    def from_binary(self, b):
        return self._from_meta_list(struct.unpack(self.BINARY_FORMAT, b))

    def to_binary(self):
        vs = []
        for field in self.FIELDS:
            t = field['type']
            v = getattr(self, self._internal_name(field['name']))
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
            t = field['type']
            _name = self._internal_name(field['name'])
            if t == 'short_string' or t == 'long_string':
                iv_next = iv + self.INTERNAL_LEN_FROM_TYPE[t]
                setattr(self, _name, b''.join(vs[iv:iv_next]).decode('utf-8'))
                iv = iv_next
            else:
                setattr(self, _name, vs[iv])
                iv += 1
        return self

    @classmethod
    def _logical_from_internal(cls, n):
        if n == cls.UNDEFINED_LOGICAL:
            return None
        else:
            return n == 1

    @classmethod
    def _internal_from_logical(cls, l):
        if l is None:
            return cls.UNDEFINED_LOGICAL
        else:
            if l:
                return 1
            else:
                return 0

    @classmethod
    def _integer_from_internal(cls, n):
        if n == cls.UNDEFINED_INTEGER:
            return None
        else:
            return n

    @classmethod
    def _internal_from_integer(cls, n):
        if n is None:
            return cls.UNDEFINED_INTEGER
        else:
            assert INTEGER_MIN <= n <= INTEGER_MAX
            return n

    @classmethod
    def _float_from_internal(cls, x):
        if x == cls.UNDEFINED_FLOAT:
            return None
        else:
            return x

    @classmethod
    def _internal_from_float(cls, x):
        if x is None:
            return cls.UNDEFINED_FLOAT
        else:
            assert FLOAT_MIN <= x <= FLOAT_MAX
            return x

    @classmethod
    def _enum_from_internal(cls, n):
        if n == cls.UNDEFINED_ENUM:
            return None
        else:
            assert 1 <= n <= len(cls.ENUMS)
            return cls.ENUMS[n - 1]

    @classmethod
    def _internal_from_enum(cls, s):
        if s is None:
            return cls.UNDEFINED_ENUM
        else:
            return cls.ENUMS.index(s) + 1

    @classmethod
    def _short_string_from_internal(cls, s):
        if s == cls.UNDEFINED_SHORT_STRING:
            return None
        else:
            return s.rstrip()

    @classmethod
    def _internal_from_short_string(cls, s):
        if s is None:
            return cls.UNDEFINED_SHORT_STRING
        else:
            assert len(s) <= cls.LEN_SHORT_STRING
            return cls.ASCII_FORMAT_FROM_TYPE['short_string'].format(s)

    @classmethod
    def _long_string_from_internal(cls, s):
        if s == cls.UNDEFINED_LONG_STRING:
            return None
        else:
            return s.rstrip()

    @classmethod
    def _internal_from_long_string(cls, s):
        if s is None:
            return cls.UNDEFINED_LONG_STRING
        else:
            assert len(s) <= cls.LEN_LONG_STRING
            return cls.ASCII_FORMAT_FROM_TYPE['long_string'].format(s)

    @staticmethod
    def _internal_name(s):
        return '_' + s

    @classmethod
    def _make_from_ascii(cls):
        _name_il_ir_fns = []
        il = 0
        for field in cls.FIELDS:
            t = field['type']
            ir = il + cls.ASCII_WIDTH_FROM_TYPE[t]
            fn = cls.PARSE_ASCII_FROM_TYPE[t]
            _name_il_ir_fns.append((cls._internal_name(field['name']), il, ir, fn))
            il = ir
            if field['eol']:
                ir = il + 1
                il = ir
        ascii_byte = ir

        def from_ascii(self, s):
            assert len(s.encode()) == ascii_byte
            for _name, il, ir, fn in _name_il_ir_fns:
                setattr(self, _name, fn(s[il:ir]))
            return self
        return from_ascii

Meta.from_ascii = Meta._make_from_ascii()
for name in Meta.NAMES:
    def make_property(name):
        type_ = Meta.TYPE_FROM_NAMES[name]
        _name = Meta._internal_name(name)
        from_ = '_{}_from_internal'.format(type_)
        to_ = '_internal_from_{}'.format(type_)
        return property(lambda self: getattr(self, from_)(getattr(self, _name)),
                        lambda self, value: setattr(self, _name, getattr(self, to_)(value)))
    setattr(Meta, name, make_property(name))

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
            self.assertAlmostEqual(self.h._delta, Meta.UNDEFINED_FLOAT)
            self.assertTrue(self.h.delta is None)
            with self.assertRaises(AssertionError):
                self.h.delta = 2*FLOAT_MIN
            with self.assertRaises(AssertionError):
                self.h.delta = 2*FLOAT_MAX

            self.h.nzyear = 2
            self.assertEqual(self.h._nzyear, 2)
            self.assertEqual(self.h.nzyear, 2)
            self.h.nzyear = None
            self.assertEqual(self.h._nzyear, Meta.UNDEFINED_INTEGER)
            self.assertEqual(self.h.nzyear, None)
            with self.assertRaises(AssertionError):
                self.h.nzyear = 2*INTEGER_MIN
            with self.assertRaises(AssertionError):
                self.h.nzyear = 2*FLOAT_MAX

            self.h.iftype = 'it'
            self.assertEqual(self.h._iftype, 85)
            self.assertEqual(self.h.iftype, 'it')
            self.h.iftype = None
            self.assertEqual(self.h._iftype, Meta.UNDEFINED_ENUM)
            self.assertEqual(self.h.iftype, None)
            with self.assertRaises(Exception):
                self.h.iftype = 'not_member_of_enums'
            self.h._iftype = 0
            with self.assertRaises(AssertionError):
                self.h.iftype
            self.h._iftype = len(Meta.ENUMS) + 1
            with self.assertRaises(Exception):
                self.h.iftype

            self.h.leven = True
            self.assertEqual(self.h._leven, 1)
            self.assertEqual(self.h.leven, True)
            self.h.leven = False
            self.assertEqual(self.h._leven, 0)
            self.assertEqual(self.h.leven, False)
            self.h.leven = None
            self.assertEqual(self.h._leven, Meta.UNDEFINED_LOGICAL)
            self.assertEqual(self.h.leven, None)

            self.h.kstnm = 'erm'
            self.assertEqual(self.h._kstnm, 'erm     ')
            self.assertEqual(self.h.kstnm, 'erm')
            self.h.kstnm = ''
            self.assertEqual(self.h._kstnm, '        ')
            self.assertEqual(self.h.kstnm, '')
            self.h.kstnm = None
            self.assertEqual(self.h._kstnm, Meta.UNDEFINED_SHORT_STRING)
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
            self.assertEqual(self.h._kevnm, Meta.UNDEFINED_LONG_STRING)
            self.assertEqual(self.h.kevnm, None)
            with self.assertRaises(AssertionError):
                self.h.kevnm = '0123456789abcdefg'

    unittest.main()
