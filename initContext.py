import sys
import os


def loadModules():
    sys.path.append(os.getcwd()+"/config")


if __name__ == '__main__':
    loadModules()
