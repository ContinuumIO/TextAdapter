from __future__ import print_function, division, absolute_import

import unittest

def test_all(descriptions=True, buffer=True, verbosity=2, failfast=False):
    loader = unittest.TestLoader()
    suite = loader.discover('.')
    runner = unittest.TextTestRunner(descriptions=descriptions,
                                     verbosity=verbosity,
                                     buffer=buffer,
                                     failfast=failfast)
    return runner.run(suite)


if __name__=='__main__':
    test_all()
