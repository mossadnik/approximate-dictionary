import numpy as np
import numba


@numba.njit
def get_symbol_bitmaps(pattern):
    res = dict()
    for i, p in enumerate(pattern):
        if not p in res:
            res[p] = np.int64(0)
        res[p] |= 1 << i
    return res


@numba.njit
def initialize_nfa(k):
    """create array with initial states"""
    res = np.empty(k + 1, dtype=np.int64)
    res[0] = 1
    for i in range(1, k + 1):
        res[i] = (2**i - 1) << 1  # 1^i0
    return res
