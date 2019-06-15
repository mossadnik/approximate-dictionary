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
from approximate_dictionary import ApproximateDictionary

strings = ['anneals', 'annual', 'bet', 'robe']
dictionary = ApproximateDictionary(strings)

hits = dictionary.search('robot', max_edits=2)

for idx in hits:
    print(strings[idx])


'robe'
```
