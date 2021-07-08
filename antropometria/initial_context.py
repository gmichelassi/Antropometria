import sys
import os


def set_context():
    sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
