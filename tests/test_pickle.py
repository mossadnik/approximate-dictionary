"""Test integration with pickle."""
# pylint: disable=redefined-outer-name,missing-docstring

import pickle
import io
import pytest

import numpy as np

from approximate_dictionary import create_index


@pytest.fixture(params=['trie', 'fb-trie'])
def indexed_strings(request):
    strings = [
        'anneal',
        'annualy',
        'but',
        'bat',
        'robot',
    ]
    method = request.param
    return strings, create_index(strings, method)

# pylint: disable=protected-access
def test_pickle(indexed_strings):
    """Test that pickling is possible and state is recovered."""
    strings, index = indexed_strings
    buffer = io.BytesIO()
    pickle.dump(index, buffer)

    buffer.seek(0)
    loaded = pickle.load(buffer)

    kwargs = dict(max_edits=4, return_distances=True)
    for s in strings:
        assert loaded.search(s, **kwargs) == index.search(s, **kwargs)
