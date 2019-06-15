import numpy as np

from approximate_dictionary import nfa


def test_get_symbol_bitmaps():
    pattern = np.array([1, 1, 2, 3], dtype=np.int32)
    bitmaps = nfa.get_symbol_bitmaps(pattern)
    # check keys are unique symbols
    assert set(bitmaps.keys()) == set(pattern)
    # check bitmaps
    for k, b in zip([1, 2, 3], ['0011', '0100', '1000']):
        assert np.binary_repr(bitmaps[k], width=4) == b
