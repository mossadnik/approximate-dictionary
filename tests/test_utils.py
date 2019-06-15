import approximate_dictionary.utils as ut


def test_array_encode_roundtrip():
    s = 'abcdefg'
    assert s == ut.array_decode(ut.array_encode(s))
