# -*- coding: UTF-8 -*-

import os, errno
import pathlib


def compute_root_dir():
    root_dir = os.path.abspath(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(__file__))))
    return root_dir + os.path.sep


proj_root_dir = pathlib.Path(compute_root_dir())


def file_exists(file_path):
    return os.path.exists(file_path)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5 (except OSError, exc: for Python <2.5)
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def make_sure_dir(dir_path):
    if not file_exists(dir_path):
        mkdir_p(dir_path)


if __name__ == "__main__":
    # print(proj_root_dir)
    pass
