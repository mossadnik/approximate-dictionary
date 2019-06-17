"""Approximate dictionary search"""

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
        """Search query string in dictionary."""
        pattern = ut.array_encode(query)
        if max_edits == 0:
            # exact search
            node = self._search_exact(
                pattern,
                self._parent_edge_ptr,
                self._edges,
                self._children)
            return self._records.get(node, None)
        # approximate search
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
    def _search_approx(pattern, edge_ptr, edges, children, depth, records, max_edits):
        # initialize NFA
        matcher = nfa.NFA(pattern, depth, max_edits)
        # store tuples of (NFA state, Trie state) for simultaneous search
        start_node = 0
        stack = [
            (0, idx_edge)
            for idx_edge in range(edge_ptr[start_node + 1] - 1, edge_ptr[start_node] - 1, -1)
        ]
        # results
        result = []
        # search
        while stack:
            idx_nfa, idx_edge = stack.pop()
            symbol = edges[idx_edge]
            is_match = matcher.process_symbol(symbol, idx_nfa)
            # check match / push new nodes
            if matcher.accepts(idx_nfa + 1):
                # check match
                node = children[idx_edge]
                if is_match:
                    record_id = records.get(node, -1)
                    if record_id >= 0:
                        result.append(record_id)
                # expand frontier
                stack += [
                    (idx_nfa + 1, idx_edge)
                    for idx_edge in range(edge_ptr[node + 1] - 1, edge_ptr[node] - 1, -1)
                ]
        return result

    def __contains__(self, s):
        return self.search(s) is not None
