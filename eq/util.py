#!/usr/bin/python


import unittest
# import sys
import functools
# import re
# import pickle

import numpy as np
# import scipy as sp
# import matplotlib.pyplot as plt
# import pandas as pd


def dots(*ms):
    return functools.reduce(np.dot, ms)


class _Tester(unittest.TestCase):

    def test_(self):
        pass


if __name__ == '__main__':
    unittest.main()
