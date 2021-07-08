import os
import sys


def set_tests_context():
    sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
