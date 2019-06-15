import pytest
import Levenshtein

from approximate_dictionary import ApproximateDictionary


@pytest.fixture
def dictionary():
    strings = [
        'anneal',
        'annualy',
        'but',
        'bat'
        'robot',
    ]
    return strings, ApproximateDictionary(strings)


def test_search(dictionary):
    strings, trie = dictionary
    for q in ['anneal', 'bet']:
        # exact search
        assert (q in trie) == (q in strings)
        distances = [(i, Levenshtein.distance(q, s)) for i, s in enumerate(strings)]
        for k in range(1, 4):
            actual = trie.search(q, max_edits=k)
            expected = [i for i, d in distances if d <= k]
            assert set(actual) == set(expected)
