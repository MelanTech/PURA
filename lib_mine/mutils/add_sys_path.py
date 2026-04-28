import os
import sys


def add_sys_path(path):
    if path not in sys.path:
        sys.path.append(path)
