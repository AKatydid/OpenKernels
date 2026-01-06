import torch
import torch.nn as nn
import cupy as cp
from cupyx.profiler import benchmark

cumsum_kernel_v1 = """
"""

cumsum_v1 = cp.RawKernel(cumsum_kernel_v1, 'cumsum_kernel')


def torch_cumsum(x):
    return torch.cumsum(x, dim=1)


def main():
    pass
