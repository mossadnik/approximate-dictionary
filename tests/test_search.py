import pytest
import Levenshtein

from approximate_dictionary import ForwardBackwardTrie


@pytest.fixture
def strings():
    return [
        'anneal',
        'annualy',
        'but',
        'bat',
        'robot',
    ]


def test_search(strings):
    dictionary = ForwardBackwardTrie.build(strings)
    for q in ['anneal', 'bet']:
        # exact search
        assert (q in dictionary) == (q in strings)
        distances = [(i, Levenshtein.distance(q, s)) for i, s in enumerate(strings)]
        for max_edits in range(1, 4):
            actual = dictionary.search(q, max_edits=max_edits)
            expected = [i for i, d in distances if d <= max_edits]
            assert actual == set(expected)
