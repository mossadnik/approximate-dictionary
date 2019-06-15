import numpy as np
import numba

from . import utils as ut
from . import nfa


class ApproximateDictionary:
    def __init__(self, strings):
        data = sorted(enumerate(strings), key=lambda x: x[1])
        self._parent_edge_ptr, self._edges, self._children, self._records = self._build(data)
        self.depth = 1 + max(len(s) for s in strings)

    def _build(self, data):
        # build trie in sparse adjacency form
        row_data = []
        path = [0]  # stores nodes for current word
        last_s = np.array([], dtype=np.int32)
        node_count = 1
        records = numba.typed.Dict()

        for record_id, s_ in data:
            s = ut.array_encode(s_)
            start = ut.common_prefix_length(s, last_s)
            last_s = s
            parent = path[start]
            path = path[:start + 1]

            for c in s[start:]:
                row_data.append((parent, c, node_count))
                path.append(node_count)
                parent = node_count
                node_count += 1
            records[np.int32(node_count - 1)] = record_id

        row_data.sort()
        row_data = np.array(row_data, dtype=np.int32)

        # create compressed parent-edge list, edge symbols and child node arrays
        vals, counts = np.unique(row_data[:, 0], return_counts=True)
        parent_edge_ptr = np.zeros(node_count + 1, dtype=np.int32)
        parent_edge_ptr[vals + 1] = counts
        parent_edge_ptr = np.cumsum(parent_edge_ptr)
        edges = row_data[:, 1]
        children = row_data[:, 2]
        return parent_edge_ptr, edges, children, records

    def search(self, query, max_edits=0):
        pattern = ut.array_encode(query)
        if max_edits == 0:
            # exact search
            node = self._search_exact(
                pattern,
                self._parent_edge_ptr,
                self._edges,
                self._children)
            return self._records.get(node, None)
        else:
            return self._search_approx(
                pattern,
                self._parent_edge_ptr,
                self._edges,
                self._children,
                self.depth,
                self._records,
                max_edits)

    @staticmethod
    @numba.njit
    def _search_exact(pattern, edge_ptr, edges, children):
        node = 0
        for c in pattern:
            lo, hi = edge_ptr[node:node + 2]
            idx = lo + np.searchsorted(edges[lo:hi], c, side='left')
            if idx >= hi or c != edges[idx]:
                return -1
            node = children[idx]
        return node

    @staticmethod
    @numba.njit
    def _search_approx(pattern, edge_ptr, edges, children, depth, records, k):
        # initialize NFA
        bitmaps = nfa.get_symbol_bitmaps(pattern)
        state_size = k + 1
        state = np.zeros((depth, state_size), dtype=np.int64)

        state[0] = nfa.initialize_nfa(k)
        first_active = np.zeros(depth, dtype=state.dtype)

        # stores tuples of (NFA state, Trie state)
        stack = [(0, idx_edge) for idx_edge in range(edge_ptr[1] - 1, edge_ptr[0] - 1, -1)]
        # results
        check = 1 << pattern.size
        result = []
        # search
        while len(stack) > 0:
            idx_nfa, idx_edge = stack.pop()
            symbol = edges[idx_edge]
            B = bitmaps.get(symbol, np.int64(0))

            # get NFA state
            old_state, new_state = state[idx_nfa], state[idx_nfa + 1]
            old_first_active = first_active[idx_nfa]
            new_first_active = old_first_active

            # process symbol
            is_match = False
            new_state[:old_first_active] = 0
            for i in range(old_first_active, state_size):
                new_state[i] = (old_state[i] & B) << 1
                if i > 0:
                    new_state[i] |= (
                        old_state[i - 1]
                        | (old_state[i - 1] << 1)
                        | (new_state[i - 1] << 1))

                if new_state[i] == 0:
                    new_first_active += 1
                else:
                    is_match |= (new_state[i] & check) != 0

            # check match / push new nodes
            if new_first_active < state_size:
                # check match
                node = children[idx_edge]
                if is_match:
                    record_id = records.get(node, -1)
                    if record_id >= 0:
                        result.append(record_id)
                # expand frontier
                stack += [
                    (idx_nfa + 1, idx_edge)
                    for idx_edge in range(edge_ptr[node + 1] - 1, edge_ptr[node] - 1, -1)]
                first_active[idx_nfa + 1] = new_first_active
        return result

    def __contains__(self, s):
        return self.search(s) is not None
