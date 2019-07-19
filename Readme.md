# Dictionary with approximate search

 * Fast approximate search for a pattern in a list of strings
 * Limited to short patterns up to 126 characters (depends on method used)
 * Uses [numba](https://numba.pydata.org), so first call can be slow

## Quickstart

Install from github

```
pip install https://github.com/mossadnik/approximate-dictionary
```

Basic usage:

```python
from approximate_dictionary import create_index

strings = ['anneals', 'annual', 'bet', 'robe']
dictionary = create_index(strings, method='fb-trie')

# Returns dict of { list index -> edit distance } for matching strings
# If return_distances is False, returns a set of indices instead.
hits = dictionary.search('robot', max_edits=2, return_distances=True)


# To recover strings, they have to be kept as well
for idx in hits:
    print(strings[idx])


'robe'
```
