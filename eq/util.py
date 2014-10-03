import unittest
import functools

import numpy as np


def dots(*ms):
    return functools.reduce(np.dot, ms)


class _Tester(unittest.TestCase):

    def test_(self):
        pass


if __name__ == '__main__':
    unittest.main()
