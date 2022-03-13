"""Implementation of Unsupervised Second-order Hidden Morkov Model using multiprocessing."""
"""二阶马尔可夫模型的无监督实现
代码参考自：https://github.com/riyazbhat/Unsupervised-Second-Order-HMM
"""
__Author__ = "Riyaz Ahmad Bhat"
__Version__ = "1.0"

import sys
import ctypes
import logging
import warnings
import numpy as np
from time import time
from itertools import izip_longest
from multiprocessing import Process, Array, cpu_count, current_process, Queue
warnings.filterwarnings("ignore")
