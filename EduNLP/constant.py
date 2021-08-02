# coding: utf-8
# 2021/8/1 @ tongshiwei

import os
from os.path import expanduser, join

ROOT = os.environ.get("EDUNLPPATH", join(expanduser("~"), ".EduNLP"))
MODEL_DIR = os.environ.get("EDUNLPMODELPATH", join(ROOT, "model"))
