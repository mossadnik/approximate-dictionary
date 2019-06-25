import pytest
import Levenshtein

from approximate_dictionary import create_index


@pytest.fixture(params=['fb-trie', 'trie'])
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


def test_search(indexed_strings):
    strings, dictionary = indexed_strings
    for q in ['anneal', 'bet']:
        # exact search
        assert (len(dictionary.search(q)) > 0) == (q in strings)
        distances = [(i, Levenshtein.distance(q, s)) for i, s in enumerate(strings)]
        for max_edits in range(1, 4):
            actual = dictionary.search(q, max_edits=max_edits)
            expected = [i for i, d in distances if d <= max_edits]
            assert actual == set(expected)
