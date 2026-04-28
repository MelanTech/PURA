from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)

prj_path = osp.normpath(osp.join(this_dir, '../..'))
add_path(prj_path)

# 将库加入环境变量
add_path(osp.join(prj_path, 'lib_mine'))
