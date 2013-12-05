import distutils.core

import eq


distutils.core.setup(name='eq',
                     version=eq.__version__,
                     author='kshramt',
                     author_email='thisisdummy@example.com',
                     url='http://www.thisisdummy.example.com',
                     description='Earthquake science library.',
                     long_description='Handy utilities for earthquake science.',
                     packages=['eq'],
                     classifiers=["Development Status :: 1 - Planning",
                                  "Environment :: Console",
                                  "Intended Audience :: Education",
                                  "Intended Audience :: Science/Research",
                                  "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
                                  "Operating System :: Unix",
                                  "Programming Language :: Python :: 3",
                                  "Topic :: Scientific/Engineering",],
                     requires=['matplotlib',
                               "kshramt"])
