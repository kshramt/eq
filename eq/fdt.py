"""
[FDT format](http://web.archive.org/web/20141002074925/http://www.kueps.kyoto-u.ac.jp/~web-bs/tsg/software/FDTformat/)
"""


def load(fp):
    return (parse_record(line.rstrip('\n')) for line in fp)


def parse_record(line):
    f_az, f_pl, s_az, s_pl, faulting_type, *comment = line.split(maxsplit=5)
    faulting_type = faulting_type.lower()
    assert faulting_type in ('n', 'r', 'd', 's')
    ret = dict(f_az=float(f_az), f_pl=float(f_pl), s_az=float(s_az), s_pl=float(s_pl), faulting_type=faulting_type)
    if comment:
        ret['comment'] = comment[0]
    return ret


def test():
    assert parse_record('10.0 -20 30 40 N') == {'f_az': 10.0, 'f_pl': -20.0, 's_az': 30.0, 's_pl': 40.0, 'faulting_type': 'n'}
    assert parse_record('10.0 -20 30 40 N a comment') == {'f_az': 10.0, 'f_pl': -20.0, 's_az': 30.0, 's_pl': 40.0, 'faulting_type': 'n', 'comment': 'a comment'}


if __name__ == '__main__':
    test()
