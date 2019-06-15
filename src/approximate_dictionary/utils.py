import numpy as np
import numba


def array_encode(s):
    """encode string into numpy.ndarray using utf-32."""
    return np.frombuffer(s.encode('utf32'), dtype=np.int32, offset=4)


def array_decode(arr):
    """decode numpy.ndarray into string"""
    if arr.dtype != np.int32:
        raise ValueError('Incompatible dtype: expected numpy.int32, got numpy.%s' % arr.dtype)
    return bytes(arr).decode('utf32')


@numba.njit
def common_prefix_length(s, u):
    """Find length of longest common prefix of two sequences."""
    length = 0
    for cs, cu in zip(s, u):
        if cs != cu:
            break
        length += 1
    return length
