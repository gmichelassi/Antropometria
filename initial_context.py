import sys
import os


def load_modules():
    sys.path.append(f"{os.getcwd()}/config")


if __name__ == '__main__':
    load_modules()
