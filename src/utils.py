import os
import random

import tensorflow as tf
import torch
import numpy as np

from choose_low_utility_gpu import choose_low_utility_gpu


def environment_set(seed: int = 42, limit: int = 5000):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(choose_low_utility_gpu(limit))
    torch.manual_seed(seed)
    tf.random.set_seed(seed)

    seed %= (1 << 31) - 1
    np.random.seed(seed)
    random.seed(seed)
