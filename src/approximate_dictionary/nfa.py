"""Row-wise bit parallel NFA for Levenshtein distance."""

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


@numba.jitclass([
    ('bitmaps', numba.types.DictType(numba.int32, numba.int64)),
    ('state', numba.int64[:, :]),
    ('first_active', numba.int64[:]),
    ('_check', numba.int64),
    ('max_edits', numba.int64),
])
class NFA:
    def __init__(self, pattern, max_states, max_edits):
        self.bitmaps = get_symbol_bitmaps(pattern)
        self.state = np.zeros((max_states, max_edits + 1), dtype=np.int64)
        self.state[0] = initialize_nfa(max_edits)
        self.first_active = np.zeros(max_states, dtype=np.int64)
        self._check = 1 << pattern.size
        self.max_edits = max_edits

    def process_symbol(self, symbol, idx_state):
        symbol_bitmap = self.bitmaps.get(symbol, np.int64(0))

        # get NFA state
        old_state, new_state = self.state[idx_state], self.state[idx_state + 1]
        old_first_active = self.first_active[idx_state]
        new_first_active = old_first_active

        # process symbol
        is_match = False
        new_state[:old_first_active] = 0
        for i in range(old_first_active, self.max_edits + 1):
            new_state[i] = (old_state[i] & symbol_bitmap) << 1
            if i > 0:
                new_state[i] |= (
                    old_state[i - 1]
                    | (old_state[i - 1] << 1)
                    | (new_state[i - 1] << 1))

            if new_state[i] == 0:
                new_first_active += 1
            else:
                is_match |= (new_state[i] & self._check) != 0
        self.first_active[idx_state + 1] = new_first_active
        return is_match

    def accepts(self, idx_state):
        return self.first_active[idx_state] <= self.max_edits
