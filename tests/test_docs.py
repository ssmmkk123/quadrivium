"""Run the doctests in the package docstrings.

Documentation that is never executed drifts silently out of date; this keeps
the quick-start examples honest.
"""

import doctest
import unittest

import numethods


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(numethods))
    return tests


if __name__ == "__main__":
    unittest.main(verbosity=2)
