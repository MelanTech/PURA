import os


def get_hostname():
    return os.environ.get("hostname")
