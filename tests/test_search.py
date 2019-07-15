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


@pytest.mark.parametrize(
    'return_distances', [True, False]
)
def test_search(indexed_strings, return_distances):
    strings, dictionary = indexed_strings
    kwargs = dict(return_distances=return_distances)
    for q in ['anneal', 'bet']:
        # exact search
        assert (len(dictionary.search(q, **kwargs)) > 0) == (q in strings)
        distances = [(i, Levenshtein.distance(q, s)) for i, s in enumerate(strings)]
        for max_edits in range(1, 4):
            actual = dictionary.search(q, max_edits=max_edits, **kwargs)
            expected = {i: d for i, d in distances if d <= max_edits}
            if return_distances:
                assert actual == expected
            else:
                assert actual == set(expected.keys())
