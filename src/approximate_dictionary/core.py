"""Trie data structure"""

import numpy as np
import numba

from . import utils as ut
from . import nfa


def create_trie(strings):
    """Build trie from sequence of strings."""
    row_data = []
    path = [0]  # stores nodes for current word
    last_s = ut.array_encode('')
    node_count = 1
    records = numba.typed.Dict()
    depth = 0

    for record_id, s_ in sorted(enumerate(strings), key=lambda x: x[1]):
        s = ut.array_encode(s_)
        start = ut.common_prefix_length(s, last_s)
        last_s = s
        parent = path[start]
        path = path[:start + 1]
        depth = max(depth, s.size + 1)

        for c in s[start:]:
            row_data.append((parent, c, node_count))
            path.append(node_count)
            parent = node_count
            node_count += 1
        records[np.int32(node_count - 1)] = np.int32(record_id)

    row_data.sort()
    row_data = np.array(row_data, dtype=np.int32)

    # create compressed parent-edge list, edge symbols and child node arrays
    vals, counts = np.unique(row_data[:, 0], return_counts=True)
    edge_ptr = np.zeros(node_count + 1, dtype=np.int32)
    edge_ptr[vals + 1] = counts
    edge_ptr = np.cumsum(edge_ptr, dtype=np.int32)
    edges = row_data[:, 1]
    children = row_data[:, 2]
    return Trie(edge_ptr, edges, children, records, depth)


@numba.jitclass([
    ('edge_ptr', numba.int32[:]),
    ('edges', numba.int32[:]),
    ('children', numba.int32[:]),
    ('records', numba.types.DictType(numba.int32, numba.int32)),
    ('depth', numba.int64),
])
class Trie:
    """Trie data structure."""
    def __init__(self, edge_ptr, edges, children, records, depth):
        self.edge_ptr = edge_ptr
        self.edges = edges
        self.children = children
        self.records = records
        self.depth = depth

    def search(self, pattern):
        """Exact pattern search."""
        edge_ptr, edges, children = self.edge_ptr, self.edges, self.children
        node = np.int32(0)
        failed = np.int32(-1)
        for c in pattern:
            lo, hi = edge_ptr[node:node + 2]
            idx = lo + np.searchsorted(edges[lo:hi], c, side='left')
            if idx >= hi or c != edges[idx]:
                return set([np.int32(0) for i in range(0)])
            node = children[idx]
        res = np.int32(self.records.get(node, failed))
        if res == -1:
            return set([np.int32(0) for i in range(0)])
        return set([res])


    def iter_matches(self, start_node, matcher):
        """Iterate over matching nodes.

        Yields
        ------
        node, distance : tuple of int
        """
        edge_ptr, edges, children = self.edge_ptr, self.edges, self.children
        start_node = np.int32(start_node)
        # store tuples of (NFA state, Trie state) for simultaneous search
        stack = [
            (0, idx_edge)
            for idx_edge in range(edge_ptr[start_node + 1] - 1, edge_ptr[start_node] - 1, -1)
        ]
        # check root node (delete all symbols from pattern)
        distance = matcher.get_distance(0)
        if distance != -1:
            yield start_node, distance
        # search
        while stack:
            idx_nfa, idx_edge = stack.pop()
            symbol = edges[idx_edge]
            matcher.process_symbol(symbol, idx_nfa)
            # check match / push new nodes
            if matcher.is_active(idx_nfa + 1):
                # check match
                node = children[idx_edge]
                distance = matcher.get_distance(idx_nfa + 1)
                if distance != -1:
                    yield node, distance
                # expand frontier
                stack += [
                    (idx_nfa + 1, idx_edge)
                    for idx_edge in range(edge_ptr[node + 1] - 1, edge_ptr[node] - 1, -1)
                ]


@numba.njit
def trie_search(trie, pattern, max_edits):
    matcher = nfa.NFA(pattern, trie.depth, max_edits)
    res = set([np.int32(0) for _ in range(0)])
    for node, distance in trie.iter_matches(np.int32(0), matcher):
        record = trie.records.get(node, np.int32(-1))
        if record != -1:
            res.add(np.int32(record))
    return res


@numba.njit
def _two_step_search(trie, head, tail,
        max_edits, max_edits_head):
    """Two-step search for FBTrie algorithm"""
    # initialize NFAs
    head_matcher = nfa.NFA(head, trie.depth, max_edits_head)
    tail_matcher = nfa.NFA(tail, trie.depth, max_edits)
    # see https://numba.pydata.org/numba-doc/dev/user/troubleshoot.html, section on untyped lists
    res = set([np.int32(0) for _ in range(0)])
    for node_head, distance_head in trie.iter_matches(np.int32(0), head_matcher):
        tail_matcher.max_edits = max_edits - distance_head
        for node_tail, distance_tail in trie.iter_matches(node_head, tail_matcher):
            record = trie.records.get(node_tail, np.int32(-1))
            if record != -1:
                res.add(np.int32(record))
    return res


@numba.njit
def fbtrie_search(forward_trie, backward_trie, pattern, max_edits):
    """Approximate search with FBTrie algorithm."""
    split = pattern.size // 2
    head, tail = pattern[:split], pattern[split:]
    max_edits_head = int(np.ceil(max_edits / 2)) - 1
    max_edits_head_rev = int(np.floor(max_edits / 2))
    res = _two_step_search(forward_trie, head, tail, max_edits=max_edits, max_edits_head=max_edits_head)
    res.update(
        _two_step_search(backward_trie, tail[::-1], head[::-1], max_edits=max_edits, max_edits_head=max_edits_head_rev)
    )
    return res
