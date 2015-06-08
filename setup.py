import distutils.core

import eq


distutils.core.setup(name='eq',
                     version=eq.__version__,
                     author='kshramt',
                     author_email='thisisdummy@example.com',
                     url='https://github.com/kshramt/eq',
                     description='Earthquake science library.',
                     long_description='Handy utilities for earthquake science.',
                     packages=['eq'],
                     classifiers=["License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
                                  "Operating System :: Unix",
                                  "Programming Language :: Python :: 3"],
                     requires=[
                         'numpy',
                     ])
