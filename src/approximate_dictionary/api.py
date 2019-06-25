"""Approximate dictionary search"""

import numpy as np
import numba

from . import utils as ut
from . import nfa
from . import trie


class ForwardBackwardTrie:
    def __init__(self, forward_trie, backward_trie):
        """Constructor for internal use only. Use ForwardBackwardTrie.build instead."""
        self.forward_trie = forward_trie
        self.backward_trie = backward_trie

    @classmethod
    def build(cls, strings):
        """Create approximate dictionary search structure from list of strings.

        Parameters
        ----------
        strings : sequence of str
            Strings to index.

        Returns
        ForwardBackwardTrie
        """
        forward_trie = trie.create_trie(strings)
        backward_trie = trie.create_trie([s[::-1] for s in strings])
        return cls(forward_trie, backward_trie)

    def search(self, s, max_edits=0):
        """Search dictionary.

        Note that only the indices of the matching strings are returned, so that
        the string list passed to ForwardBackwardTrie.build has to be kept if the
        matching strings themselves are required.

        Parameters
        ----------
        s : str
            query string
        max_edits : int, optional
            maximum edit distance, defaults to zero.

        Returns
        -------
        hits : set of int
            Sequence indices of the strings that match the query.
        """
        pattern = ut.array_encode(s)
        if max_edits == 0:
            return self.forward_trie.search(pattern)
        return trie.fbtrie_search(self.forward_trie, self.backward_trie, pattern, max_edits)

    def __contains__(self, s):
        return len(self.search(s)) > 0
