"""Exposed classes and functions."""

from . import utils as ut
from . import core


def create_index(strings, method='fb-trie'):
    """Create a searchable index for a list of strings.

    Parameters
    ----------
    strings : iterable of str
        Strings to index
    method : {'fb-trie', 'trie'}
        Index method to use. This affects memory usage and
        query run time.

        'fb-trie' is several times faster than 'trie' but consumes
        twice as much memory for the index. It allows query strings of length up to 126
        characters, whereas 'trie' is limited to 63.

        Both methods use tries that require around 16 bytes per node. Typically there are 2-4 characters
        per node, so that a 'trie' index is 4-8 times larger than the python strings
        in the case of ascii (1-4 for general unicode). An 'fb-trie' index is twice as large.

    Returns
    -------
    index
        Searchable index structure.
    """
    index_cls = {
        'trie': TrieIndex,
        'fb-trie': ForwardBackwardTrieIndex,
    }
    if method not in index_cls:
        raise ValueError('method must be one of ("trie", "fb-trie").')
    return index_cls[method].create_index(strings)


class TrieIndex:
    """String index based on forward Trie."""
    def __init__(self, trie):
        """Constructor for internal use only. Use Trie.build instead."""
        self._trie = trie

    @classmethod
    def create_index(cls, strings):
        """Create approximate dictionary search structure from list of strings.

        Parameters
        ----------
        strings : sequence of str
            Strings to index.

        Returns
        -------
        Trie
        """
        trie = core.create_trie(strings)
        return cls(trie)

    def search(self, s, max_edits=0, return_distances=True):
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
        hits : set or dict
            Sequence indices of the strings that match the query.
            If return_distances is set to True, returns { index -> distance }.
        """
        pattern = ut.array_encode(s)
        if max_edits == 0:
            res = {k: 0 for k in self._trie.search(pattern)}
        else:
            res = dict(core.trie_search(self._trie, pattern, max_edits))
        return res if return_distances else set(res.keys())


class ForwardBackwardTrieIndex:
    """FBTrie method for approximate dictionary search.

    This method uses about twice as much memory as the Trie method, but is several times faster.

    see
    Boytsov 2011: Indexing Methods for Approximate Dictionary Searching: Comparative Analysis
    """
    def __init__(self, forward_trie, backward_trie):
        """Constructor for internal use only. Use ForwardBackwardTrieIndex.build instead."""
        self._forward_trie = forward_trie
        self._backward_trie = backward_trie

    @classmethod
    def create_index(cls, strings):
        """Create approximate dictionary search structure from list of strings.

        Parameters
        ----------
        strings : sequence of str
            Strings to index.

        Returns
        -------
        ForwardBackwardTrieIndex
        """
        forward_trie = core.create_trie(strings)
        backward_trie = core.create_trie([s[::-1] for s in strings])
        return cls(forward_trie, backward_trie)

    def search(self, s, max_edits=0, return_distances=True):
        """Search dictionary.

        Note that only the indices of the matching strings are returned, so that
        the indexed string list has to be kept if the matching strings themselves
        are required.

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
            If return_distances is set to True, returns { index -> distance }.
        """
        pattern = ut.array_encode(s)
        if max_edits == 0:
            res = {k: 0 for k in self._forward_trie.search(pattern)}
        else:
            res = dict(
                core.fbtrie_search(self._forward_trie, self._backward_trie, pattern, max_edits))
        return res if return_distances else set(res.keys())
