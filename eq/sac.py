"""
# Note

Header fields are not automatically updated.

# REFERENCES

- [SAC User Manual/SAC Data Format](http://www.iris.edu/software/sac/manual/file_format.html)
"""

import struct
import unittest
import itertools
import collections

import numpy as np


FLOAT = np.float32
INTEGER = np.int32
COMPLEX = np.complex64
_UNDEFINED_SHORT_STRING = '-12345  '
_UNDEFINED_LONG_STRING = '-12345          '
_N_BYTES_SHORT_STRING = len(_UNDEFINED_SHORT_STRING.encode())
_N_BYTES_LONG_STRING = len(_UNDEFINED_LONG_STRING.encode())
_UNDEFINED_OF_TYPE = {
    'logical': -12345,
    'integer': -12345,
    'float': -12345.0,
    'enum': -12345,
    'short_string': _UNDEFINED_SHORT_STRING,
    'long_string': _UNDEFINED_LONG_STRING,
}
_ENUM_DOCS = collections.OrderedDict((
    ('ireal', 'Undocumented'),
    ('itime', 'Time series file'),
    ('irlim', 'Spectral file-real/imag'),
    ('iamph', 'Spectral file-ampl/phase'),
    ('ixy', 'General x vs y file'),
    ('iunkn', 'Unknown'),
    ('idisp', 'Displacement (Nm)'),
    ('ivel', 'Velocity (Nm/s)'),
    ('iacc', 'Acceleration (Nm/s/s)'),
    ('ib', 'Begin time'),
    ('iday', 'GMT day'),
    ('io', 'Event origin time'),
    ('ia', 'First arrival time'),
    ('it0', 'User defined time pick 0'),
    ('it1', 'User defined time pick 1'),
    ('it2', 'User defined time pick 2'),
    ('it3', 'User defined time pick 3'),
    ('it4', 'User defined time pick 4'),
    ('it5', 'User defined time pick 5'),
    ('it6', 'User defined time pick 6'),
    ('it7', 'User defined time pick 7'),
    ('it8', 'User defined time pick 8'),
    ('it9', 'User defined time pick 9'),
    ('iradnv', 'Radial (NTS)'),
    ('itannv', 'Tangential (NTS)'),
    ('iradev', 'Radial (EVENT)'),
    ('itanev', 'Tangential (EVENT)'),
    ('inorth', 'North positive'),
    ('ieast', 'East positive'),
    ('ihorza', 'Horizontal (ARB)'),
    ('idown', 'Down positive'),
    ('iup', 'Up positive'),
    ('illlbb', 'LLL broadband'),
    ('iwwsn1', 'WWSN 15-100'),
    ('iwwsn2', 'WWSN 30-100'),
    ('ihglp', 'High-gain long-period'),
    ('isro', 'SRO'),
    ('inucl', 'Nuclear event'),
    ('ipren', 'Nuclear pre-shot event'),
    ('ipostn', 'Nuclear post-shot event'),
    ('iquake', 'Earthquake'),
    ('ipreq', 'Foreshock'),
    ('ipostq', 'Aftershock'),
    ('ichem', 'Chemical explosion'),
    ('iother', 'Other'),
    ('igood', 'Good'),
    ('iglch', 'Gliches'),
    ('idrop', 'Dropouts'),
    ('ilowsn', 'Low signal to noise ratio'),
    ('irldta', 'Real data'),
    ('ivolts', 'Velocity (volts'),
    ('ixyz', 'General XYZ (3-D) file'),
    ('imb', 'Bodywave Magnitude'),
    ('ims', 'Surface Magnitude'),
    ('iml', 'Local Magnitude'),
    ('imw', 'Moment Magnitude'),
    ('imd', 'Duration Magnitude'),
    ('imx', 'User Defined Magnitude'),
    ('ineic', 'National Earthquake Information Center'),
    ('ipdeq', ''),
    ('ipdew', ''),
    ('ipde', 'Preliminary Determination of Epicenter'),
    ('iisc', 'Internation Seismological Centre'),
    ('ireb', 'Reviewed Event Bulletin'),
    ('iusgs', 'US Geological Survey'),
    ('ibrk', 'UC Berkeley'),
    ('icaltech', 'California Institute of Technology'),
    ('illnl', 'Lawrence Livermore National Laboratory'),
    ('ievloc', 'Event Location (computer program)'),
    ('ijsop', 'Joint Seismic Observation Program'),
    ('iuser', 'The individual using SAC2000'),
    ('iunknown', 'Unknown'),
    ('iqb', 'Quarry or mine blast confirmed by quarry'),
    ('iqb1', 'Quarry or mine blast with designed shot information-ripple fired'),
    ('iqb2', 'Quarry or mine blast with observed shot information-ripple fired'),
    ('iqbx', 'Quarry or mine blast - single shot'),
    ('iqmt', 'Quarry or mining-induced events: tremors and rockbursts'),
    ('ieq', 'Earthquake'),
    ('ieq1', 'Earthquakes in a swarm or aftershock sequence'),
    ('ieq2', 'Felt earthquake'),
    ('ime', 'Marine explosion'),
    ('iex', 'Other explosion'),
    ('inu', 'Nuclear explosion'),
    ('inc', 'Nuclear cavity collapse'),
    ('io_', 'Other source of known origin'),
    ('il', 'Local event of unknown origin'),
    ('ir', 'Regional event of unknown origin'),
    ('it', 'Teleseismic event of unknown origin'),
    ('iu', 'Undetermined or conflicting information'),
    ('ieq3', 'Damaging Earthquake'),
    ('ieq0', 'Probable earthquake'),
    ('iex0', 'Probable explosion'),
    ('iqc', 'Mine collapse'),
    ('iqb0', 'Probable Mine Blast'),
    ('igey', 'Geyser'),
    ('ilit', 'Light'),
    ('imet', 'Meteroic event'),
    ('iodor', 'Odors'),
))
_ENUMS = tuple(_ENUM_DOCS.keys())
_INTEGER_MIN = np.iinfo(INTEGER).min
_INTEGER_MAX = np.iinfo(INTEGER).max
_FLOAT_MIN = np.finfo(FLOAT).min
_FLOAT_MAX = np.finfo(FLOAT).max
_IS_VALID_OF_TYPE = dict(
    logical=lambda l: isinstance(l, bool) or isinstance(l, np.bool_),
    integer=lambda n: _INTEGER_MIN <= n <= _INTEGER_MAX,
    float=lambda x: _FLOAT_MIN <= x <= _FLOAT_MAX,
    enum=lambda n: n in _ENUMS,
    short_string=lambda s: len(s.encode()) <= _N_BYTES_SHORT_STRING,
    long_string=lambda s: len(s.encode()) <= _N_BYTES_LONG_STRING,
)
_DEFAULT_OF_NAME = dict(
    nvhdr=6,
    iftype='itime',
    npts=0,
    lovrok=True,
)
_EOL_NAMES = (
    'odelta',
    'fmt',
    't4',
    't9',
    'resp3',
    'resp8',
    'stdp',
    'mag',
    'user4',
    'user9',
    'sb',
    'xminimum',
    'fhdr65',
    'fhdr70',
    'nzsec',
    'npts',
    'nhdr56',
    'iinst',
    'isynth',
    'ihdr15',
    'ihdr20',
    'lhdr5',
    'kevnm',
    'ka',
    'kt2',
    'kt5',
    'kt8',
    'kuser0',
    'kcmpnm',
    'kinst',
)
_N_BINARY_BYTES_OF_TYPE = {
    'logical': 4,
    'integer': 4,
    'float': 4,
    'enum': 4,
    'short_string': _N_BYTES_SHORT_STRING,
    'long_string': _N_BYTES_LONG_STRING,
}
_N_ASCII_BYTES_OF_TYPE = {
    'logical': 10,
    'integer': 10,
    'float': 15,
    'enum': 10,
    'short_string': _N_BYTES_SHORT_STRING,
    'long_string': _N_BYTES_LONG_STRING,
}
_BINARY_MODE = '='
_BINARY_FLOAT_FORMAT = 'f'
_BINARY_FORMAT_OF_TYPE = {
    'logical': _BINARY_MODE + 'i',
    'integer': _BINARY_MODE + 'i',
    'float': _BINARY_MODE + _BINARY_FLOAT_FORMAT,
    'enum': _BINARY_MODE + 'i',
    'short_string': _BINARY_MODE + '{}s'.format(_N_ASCII_BYTES_OF_TYPE['short_string']),
    'long_string': _BINARY_MODE + '{}s'.format(_N_ASCII_BYTES_OF_TYPE['long_string']),
}
_ASCII_FORMAT_OF_TYPE = {
    'logical': '{{:>{}d}}'.format(_N_ASCII_BYTES_OF_TYPE['logical']),
    'integer': '{{:>{}d}}'.format(_N_ASCII_BYTES_OF_TYPE['integer']),
    'float': '{{:>#{}.7g}}'.format(_N_ASCII_BYTES_OF_TYPE['float']),
    'enum': '{{:>{}d}}'.format(_N_ASCII_BYTES_OF_TYPE['enum']),
    'short_string': '{:s}',
    'long_string': '{:s}',
}
_VAL_OF_INTERNAL_OF_TYPE = dict(
    logical=lambda n: n == 1,
    enum=lambda n: _ENUMS[n],
    short_string=lambda b: b.rstrip(),
    long_string=lambda b: b.rstrip(),
)
_INTERNAL_OF_VAL_OF_TYPE = dict(
    logical=lambda l: 1 if l else 0,
    enum=lambda s: _ENUMS.index(s),
    short_string=lambda s: _pad_space(s, _N_BYTES_SHORT_STRING),
    long_string=lambda s: _pad_space(s, _N_BYTES_LONG_STRING),
)
_INTERNAL_OF_ASCII_OF_TYPE = dict(
    logical=INTEGER,
    integer=INTEGER,
    float=FLOAT,
    enum=INTEGER,
)
_FIELD_NAMES = (
    'delta',
    'depmin',
    'depmax',
    'scale',
    'odelta',
    'b',
    'e',
    'o',
    'a',
    'fmt',
    't0',
    't1',
    't2',
    't3',
    't4',
    't5',
    't6',
    't7',
    't8',
    't9',
    'f',
    'resp0',
    'resp1',
    'resp2',
    'resp3',
    'resp4',
    'resp5',
    'resp6',
    'resp7',
    'resp8',
    'resp9',
    'stla',
    'stlo',
    'stel',
    'stdp',
    'evla',
    'evlo',
    'evel',
    'evdp',
    'mag',
    'user0',
    'user1',
    'user2',
    'user3',
    'user4',
    'user5',
    'user6',
    'user7',
    'user8',
    'user9',
    'dist',
    'az',
    'baz',
    'gcarc',
    'sb',
    'sdelta',
    'depmen',
    'cmpaz',
    'cmpinc',
    'xminimum',
    'xmaximum',
    'yminimum',
    'ymaximum',
    'fhdr64', # adjtm
    'fhdr65',
    'fhdr66',
    'fhdr67',
    'fhdr68',
    'fhdr69',
    'fhdr70',
    'nzyear',
    'nzjday',
    'nzhour',
    'nzmin',
    'nzsec',
    'nzmsec',
    'nvhdr',
    'norid',
    'nevid',
    'npts',
    'nsnpts',
    'nwfid',
    'nxsize',
    'nysize',
    'nhdr56',
    'iftype',
    'idep',
    'iztype',
    'ihdr4',
    'iinst',
    'istreg',
    'ievreg',
    'ievtyp',
    'iqual',
    'isynth',
    'imagtyp',
    'imagsrc',
    'ihdr13',
    'ihdr14',
    'ihdr15',
    'ihdr16',
    'ihdr17',
    'ihdr18',
    'ihdr19',
    'ihdr20',
    'leven',
    'lpspol',
    'lovrok',
    'lcalda',
    'lhdr5',
    'kstnm',
    'kevnm',
    'khole',
    'ko',
    'ka',
    'kt0',
    'kt1',
    'kt2',
    'kt3',
    'kt4',
    'kt5',
    'kt6',
    'kt7',
    'kt8',
    'kt9',
    'kf',
    'kuser0',
    'kuser1',
    'kuser2',
    'kcmpnm',
    'knetwk',
    'kdatrd',
    'kinst',
)


def _to_ascii_of(type_, eol):
    ascii_format = _ASCII_FORMAT_OF_TYPE[type_]
    if eol:
        ascii_format += "\n"
    undefined = _UNDEFINED_OF_TYPE[type_]
    is_valid = _is_valid_of(type_)
    internal_of_val = _internal_of_val_of(type_)
    def to_ascii(x):
        if is_valid(x):
            if x is None:
                internal = undefined
            else:
                internal = internal_of_val(x)
            return ascii_format.format(internal)
        else:
            raise ValueError('invalid value {} for type {}'.format(x, type_))
    return to_ascii


def _of_ascii_of(type_, eol):
    undefined = _UNDEFINED_OF_TYPE[type_]
    internal_of_ascii = _internal_of_ascii_of(type_)
    val_of_internal = _val_of_internal_of(type_)
    def of_ascii(s):
        if eol:
            s = s[:-1]
        x = internal_of_ascii(s)
        if x != undefined:
            return val_of_internal(x)
    return of_ascii


def _to_binary_of(type_):
    binary_format = _BINARY_FORMAT_OF_TYPE[type_]
    undefined = _UNDEFINED_OF_TYPE[type_]
    is_valid = _is_valid_of(type_)
    internal_of_val = _internal_of_val_of(type_)
    if type_ in ('long_string', 'short_string'):
        encode_if_string = lambda s: s.encode()
    else:
        encode_if_string = _identity
    def to_binary(x):
        if is_valid(x):
            if x is None:
                internal = undefined
            else:
                internal = internal_of_val(x)
            return struct.pack(binary_format, encode_if_string(internal))
        else:
            raise ValueError('invalid value {} for type {}'.format(x, type_))
    return to_binary


def _of_binary_of(type_):
    binary_format = _BINARY_FORMAT_OF_TYPE[type_]
    undefined = _UNDEFINED_OF_TYPE[type_]
    val_of_internal = _val_of_internal_of(type_)
    if type_ in ('long_string', 'short_string'):
        decode_if_string = lambda s: s.decode()
    else:
        decode_if_string = _identity
    def of_binary(b):
        x = decode_if_string(struct.unpack(binary_format, b)[0])
        if x != undefined:
            return val_of_internal(x)
    return of_binary


def _internal_of_ascii_of(type_):
    return _INTERNAL_OF_ASCII_OF_TYPE.get(type_, _identity)


def _internal_of_val_of(type_):
    return _INTERNAL_OF_VAL_OF_TYPE.get(type_, _identity)


def _val_of_internal_of(type_):
    return _VAL_OF_INTERNAL_OF_TYPE.get(type_, _identity)


def _is_valid_of(type_):
    _fn = _IS_VALID_OF_TYPE[type_]
    return lambda x: x is None or _fn(x)


def _is_eol(name):
    return name in _EOL_NAMES


def _type_of(name):
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


def _default_of(name):
    return _DEFAULT_OF_NAME.get(name, None)


def _identity(x):
    return x


def _pad_space(s, n):
    nbs = len(s.encode())
    if nbs > n:
        raise ValueError('nbs > n: {}'.format(s))
    return (s + ' '*(n - nbs))


class _Field:

    def __init__(self, name):
        self.name = name
        self.type_ = _type_of(self.name)
        self.eol = _is_eol(self.name)
        self.default = _default_of(self.name)
        self.of_binary = _of_binary_of(self.type_)
        self.to_binary = _to_binary_of(self.type_)
        self.of_ascii = _of_ascii_of(self.type_, self.eol)
        self.to_ascii = _to_ascii_of(self.type_, self.eol)


_FIELDS = [_Field(name) for name in _FIELD_NAMES]


def _internal_of_data(xs, iftype):
    if iftype == 'itime':
        return _internal_of_itime(xs)
    elif iftype == 'ixy':
        return _internal_of_ixy(xs)
    elif iftype == 'iamph':
        return _internal_of_iamph(xs)
    elif iftype == 'irlim':
        return _internal_of_irlim(xs)
    else:
        raise ValueError('unsupported iftype: {}'.format(iftype))


def _internal_of_itime(xs):
    return _assert_floats(xs)


def _internal_of_ixy(xsys):
    xs, ys = xsys
    assert np.size(xs) == np.size(ys)
    return _assert_floats(np.concatenate((ys, xs)))


def _internal_of_iamph(cs):
    return _assert_floats(np.concatenate((np.absolute(cs), np.angle(cs))))


def _internal_of_irlim(cs):
    return _assert_floats(np.concatenate((np.real(cs), np.imag(cs))))


def _make_of_binary():
    name_il_ir_fns = []
    il = 0
    for field in _FIELDS:
        ir = il + _N_BINARY_BYTES_OF_TYPE[field.type_]
        name_il_ir_fns.append((field.name, il, ir, field.of_binary))
        il = ir
    n_meta_binary_bytes = ir

    def of_binary(self, b):
        nb = len(b)
        if nb < n_meta_binary_bytes:
            raise ValueError('len(b) < n_meta_binary_bytes: {}, {}'.format(len(b), n_meta_binary_bytes))
        for name, il, ir, fn in name_il_ir_fns:
            setattr(self, name, fn(b[il:ir]))
        internal = np.fromstring(b[n_meta_binary_bytes:], dtype=FLOAT)
        assert np.size(internal) == self.npts
        self.data = _data_of_internal(internal, self.iftype)
        return self
    return of_binary


def _make_of_ascii():
    name_il_ir_fns = []
    il = 0
    for field in _FIELDS:
        ir = il + _N_ASCII_BYTES_OF_TYPE[field.type_]
        if field.eol:
            ir += 1
        name_il_ir_fns.append((field.name, il, ir, field.of_ascii))
        il = ir
    n_meta_ascii_bytes = ir

    def of_ascii(self, s):
        b = s.encode()
        if len(b) < n_meta_ascii_bytes:
            raise ValueError('len(b) < n_meta_ascii_bytes: {}, {}'.format(len(b), n_meta_ascii_bytes))
        for name, il, ir, fn in name_il_ir_fns:
            setattr(self, name, fn(b[il:ir].decode()))
        internal = np.fromstring(b[n_meta_ascii_bytes + 1:].decode(), dtype=FLOAT, sep=' ')
        assert np.size(internal) == self.npts
        self.data = _data_of_internal(internal, self.iftype)
        return self
    return of_ascii


def _data_of_internal(xs, iftype):
    if iftype == 'itime':
        return _itime_of_internal(xs)
    elif iftype == 'ixy':
        return _ixy_of_internal(xs)
    elif iftype == 'iamph':
        return _iamph_of_internal(xs)
    elif iftype == 'irlim':
        return _irlim_of_internal(xs)
    else:
        raise ValueError('unsupported iftype: {}'.format(iftype))


def _itime_of_internal(xs):
    """
    ys -> ys
    """
    return xs


def _ixy_of_internal(xs):
    """
    [y1, y2, ..., x1, x2, ...] -> (xs, ys)
    """
    n = np.size(xs)
    assert n%2 == 0
    return xs[n//2:], xs[:n//2]


def _iamph_of_internal(xs):
    """
    [r1, r2, ..., θ1, θ2, ...] -> [complex(r1*cos(θ1), r1*sin(θ1)), complex(r2*cos(θ2), r2*sin(θ2)), ...]
    """
    n = len(xs)
    assert n%2 == 0
    rs = xs[:n//2]
    ts = xs[n//2:]
    return rs*np.exp(1j*ts)


def _irlim_of_internal(xs):
    """
    [r1, r2, ..., i1, i2, ...] -> [complex(r1, i1), complex(r2, i2), ...]
    """
    n = len(xs)
    assert n%2 == 0
    return xs[:n//2] + 1j*xs[n//2:]


class Sac:

    def __init__(self, x=None):
        self.of(x)

    def of(self, x):
        self._init_fields()
        if x is None:
            return self
        elif isinstance(x, dict):
            return self.of_dict(x)
        elif isinstance(x, str):
            return self.of_ascii(x)
        elif isinstance(x, bytes):
            return self.of_binary(x)
        elif hasattr(x, 'read'):
            return self.of(x.read())
        else:
            raise ValueError("unsupported input type: {}".format(type(x)))

    def __eq__(self, other):
        return bytes(self) == bytes(other)

    def __bytes__(self):
        b = b''.join(field.to_binary(getattr(self, field.name)) for field in _FIELDS)
        internal = _internal_of_data(self.data, self.iftype)
        assert np.size(internal) == self.npts
        return b + np.asarray(internal, dtype=FLOAT).tobytes()

    def __str__(self):
        s = ''.join(field.to_ascii(getattr(self, field.name)) for field in _FIELDS)
        internal = _internal_of_data(self.data, self.iftype)
        assert np.size(internal) == self.npts
        form = _ASCII_FORMAT_OF_TYPE['float']
        strs = []
        for i, x in enumerate(internal):
            strs.append(form.format(x))
            if (i + 1)%5 == 0:
                strs.append('\n')
        if len(strs) == 0 or strs[-1] != '\n':
            strs.append('\n')
        return s + ''.join(strs)

    def of_dict(self, d):
        for field in _FIELDS:
            name = field.name
            setattr(self, name, d.get(name, field.default))
        self.data = d.get('data', None)
        return self

    def to_dict(self):
        d = {field.name: getattr(self, field.name) for field in _FIELDS}
        d['data'] = self.data
        return d

    def _init_fields(self):
        for field in _FIELDS:
            setattr(self, field.name, field.default)
        self.data = []


Sac.of_binary = _make_of_binary()
Sac.of_ascii = _make_of_ascii()


def _assert_floats(xs):
    assert np.all(_FLOAT_MIN <= xs) and np.all(xs <= _FLOAT_MAX)
    return xs


class _Tester(unittest.TestCase):

    S = """\
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

    B = b'\x00\x00\x80?\x00\x00\x00\xc0\x00\x00\x80@\x00\xe4@\xc6\x00\xe4@\xc6\x00\x00\x00\x00\x00\x00\xc0@\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\x00\x80?\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\x00\xe4@\xc6\xdd\x07\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00;\x00\x00\x00;\x00\x00\x00\x01\x00\x00\x00\x06\x00\x00\x00\xc7\xcf\xff\xff\xc7\xcf\xff\xff\x07\x00\x00\x00\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\x01\x00\x00\x00\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\xc7\xcf\xff\xff\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00 \xe7\xad\x91\xe6\xb3\xa2 testing_waveform-12345  -12345  -12345  -12345  -12345  -12345  -12345  -12345  -12345  -12345  -12345  -12345  -12345  -12345  -12345  -12345  -12345  -12345  -12345  -12345  -12345  \x00\x00\x80@\x00\x00@@\x00\x00\x00@\x00\x00\x80?\x00\x00\x00\x00\x00\x00\x80\xbf\x00\x00\x00\xc0'

    D = {
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
        'npts': 7,
        'nsnpts': None,
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
        'data': [4, 3, 2, 1, 0, -1, -2],
    }

    def check_sac(self, w):
        self.assertAlmostEqual(w.delta, 1)
        self.assertAlmostEqual(w.depmin, -2)
        self.assertAlmostEqual(w.depmax, 4)
        self.assertTrue(w.scale is None)
        self.assertTrue(w.odelta is None)
        self.assertAlmostEqual(w.b, 0)
        self.assertAlmostEqual(w.e, 6)
        self.assertTrue(w.o is None)
        self.assertTrue(w.a is None)
        self.assertTrue(w.fmt is None)
        self.assertTrue(w.t0 is None)
        self.assertTrue(w.t1 is None)
        self.assertTrue(w.t2 is None)
        self.assertTrue(w.t3 is None)
        self.assertTrue(w.t4 is None)
        self.assertTrue(w.t5 is None)
        self.assertTrue(w.t6 is None)
        self.assertTrue(w.t7 is None)
        self.assertTrue(w.t8 is None)
        self.assertTrue(w.t9 is None)
        self.assertTrue(w.f is None)
        self.assertTrue(w.resp0 is None)
        self.assertTrue(w.resp1 is None)
        self.assertTrue(w.resp2 is None)
        self.assertTrue(w.resp3 is None)
        self.assertTrue(w.resp4 is None)
        self.assertTrue(w.resp5 is None)
        self.assertTrue(w.resp6 is None)
        self.assertTrue(w.resp7 is None)
        self.assertTrue(w.resp8 is None)
        self.assertTrue(w.resp9 is None)
        self.assertTrue(w.stla is None)
        self.assertTrue(w.stlo is None)
        self.assertTrue(w.stel is None)
        self.assertTrue(w.stdp is None)
        self.assertTrue(w.evla is None)
        self.assertTrue(w.evlo is None)
        self.assertTrue(w.evel is None)
        self.assertTrue(w.evdp is None)
        self.assertTrue(w.mag is None)
        self.assertTrue(w.user0 is None)
        self.assertTrue(w.user1 is None)
        self.assertTrue(w.user2 is None)
        self.assertTrue(w.user3 is None)
        self.assertTrue(w.user4 is None)
        self.assertTrue(w.user5 is None)
        self.assertTrue(w.user6 is None)
        self.assertTrue(w.user7 is None)
        self.assertTrue(w.user8 is None)
        self.assertTrue(w.user9 is None)
        self.assertTrue(w.dist is None)
        self.assertTrue(w.az is None)
        self.assertTrue(w.baz is None)
        self.assertTrue(w.gcarc is None)
        self.assertTrue(w.sb is None)
        self.assertTrue(w.sdelta is None)
        self.assertAlmostEqual(w.depmen, 1)
        self.assertTrue(w.cmpaz is None)
        self.assertTrue(w.cmpinc is None)
        self.assertTrue(w.xminimum is None)
        self.assertTrue(w.xmaximum is None)
        self.assertTrue(w.yminimum is None)
        self.assertTrue(w.ymaximum is None)
        self.assertTrue(w.fhdr64 is None)
        self.assertTrue(w.fhdr65 is None)
        self.assertTrue(w.fhdr66 is None)
        self.assertTrue(w.fhdr67 is None)
        self.assertTrue(w.fhdr68 is None)
        self.assertTrue(w.fhdr69 is None)
        self.assertTrue(w.fhdr70 is None)
        self.assertEqual(w.nzyear, 2013)
        self.assertEqual(w.nzjday, 1)
        self.assertEqual(w.nzhour, 0)
        self.assertEqual(w.nzmin, 59)
        self.assertEqual(w.nzsec, 59)
        self.assertEqual(w.nzmsec, 1)
        self.assertEqual(w.nvhdr, 6)
        self.assertTrue(w.norid is None)
        self.assertTrue(w.nevid is None)
        self.assertEqual(w.npts, 7)
        self.assertTrue(w.nsnpts is None)
        self.assertTrue(w.nwfid is None)
        self.assertTrue(w.nxsize is None)
        self.assertTrue(w.nysize is None)
        self.assertTrue(w.nhdr56 is None)
        self.assertAlmostEqual(w.iftype, 'itime')
        self.assertTrue(w.idep is None)
        self.assertTrue(w.iztype is None)
        self.assertTrue(w.ihdr4 is None)
        self.assertTrue(w.iinst is None)
        self.assertTrue(w.istreg is None)
        self.assertTrue(w.ievreg is None)
        self.assertTrue(w.ievtyp is None)
        self.assertTrue(w.iqual is None)
        self.assertTrue(w.isynth is None)
        self.assertTrue(w.imagtyp is None)
        self.assertTrue(w.imagsrc is None)
        self.assertTrue(w.ihdr13 is None)
        self.assertTrue(w.ihdr14 is None)
        self.assertTrue(w.ihdr15 is None)
        self.assertTrue(w.ihdr16 is None)
        self.assertTrue(w.ihdr17 is None)
        self.assertTrue(w.ihdr18 is None)
        self.assertTrue(w.ihdr19 is None)
        self.assertTrue(w.ihdr20 is None)
        self.assertTrue(w.leven)
        self.assertFalse(w.lpspol)
        self.assertTrue(w.lovrok)
        self.assertTrue(w.lcalda)
        self.assertFalse(w.lhdr5)
        self.assertEqual(w.kstnm, ' 筑波')
        self.assertEqual(w.kevnm, 'testing_waveform')
        self.assertTrue(w.khole is None)
        self.assertTrue(w.ko is None)
        self.assertTrue(w.ka is None)
        self.assertTrue(w.kt0 is None)
        self.assertTrue(w.kt1 is None)
        self.assertTrue(w.kt2 is None)
        self.assertTrue(w.kt3 is None)
        self.assertTrue(w.kt4 is None)
        self.assertTrue(w.kt5 is None)
        self.assertTrue(w.kt6 is None)
        self.assertTrue(w.kt7 is None)
        self.assertTrue(w.kt8 is None)
        self.assertTrue(w.kt9 is None)
        self.assertTrue(w.kf is None)
        self.assertTrue(w.kuser0 is None)
        self.assertTrue(w.kuser1 is None)
        self.assertTrue(w.kuser2 is None)
        self.assertTrue(w.kcmpnm is None)
        self.assertTrue(w.knetwk is None)
        self.assertTrue(w.kdatrd is None)
        self.assertTrue(w.kinst is None)
        self.assertEqual(len(w.data), 7)
        for parsed, orig in zip(w.data, [4, 3, 2, 1, 0, -1, -2]):
            self.assertAlmostEqual(parsed, orig)

    def test_eq_ne(self):
        for (constractor1, constractor2, w1, w2) in itertools.product(
                (Sac,),
                (Sac,),
                (self.S, self.B, self.D),
                (self.S, self.B, self.D),
        ):
            self.assertTrue(constractor1(w1) == constractor2(w2))
            self.assertFalse(constractor1(w1) != constractor2(w2))

    def test_parse_str(self):
        sac_s = Sac(self.S)
        self.check_sac(sac_s)
        self.assertEqual(str(sac_s), self.S)

    def test_parse_bytes(self):
        sac_b = Sac(self.B)
        self.check_sac(sac_b)
        self.assertEqual(bytes(sac_b), self.B)

    def test_parse_dict(self):
        sac_d = Sac(self.D)
        self.check_sac(sac_d)
        d_out = sac_d.to_dict()
        self.assertEqual(len(d_out['data']), 7)
        for parsed, orig in zip(d_out['data'], self.D['data']):
            self.assertAlmostEqual(parsed, orig)
        for field in _FIELDS:
            name = field.name
            self.assertAlmostEqual(d_out[name], self.D[name])

    # def test_ascii(self):
    #     self.assertEqual(str(self.h), str(_Meta().of_str(str(self.h))))
    #     with self.assertRaises(Error):
    #         self.h.of_str('too short')


if __name__ == '__main__':
    unittest.main()
