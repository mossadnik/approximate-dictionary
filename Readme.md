# Dictionary with approximate search

 * Fast approximate search for a pattern in a list of strings
 * Limited to short patterns up to 63 characters
 * Uses [numba](https://numba.pydata.org), so first call can be slow

## Quickstart

Install from github

```
pip install https://github.com/mossadnik/approximate_dictionary
```

Basic usage:

```python
from approximate_dictionary import ForwardBackwardTrie

strings = ['anneals', 'annual', 'bet', 'robe']
dictionary = ForwardBackwardTrie.build(strings)

# Returns set of list indices of matching strings are returned
hits = dictionary.search('robot', max_edits=2)

# To recover strings, they have to be kept as well
for idx in hits:
    print(strings[idx])


'robe'
```
