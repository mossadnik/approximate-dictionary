import approximate_dictionary.utils as ut


def test_array_encode_roundtrip():
    """test that round trip through array encode / decode
    is the identity.
    """
    s = 'abcdefg'
    assert s == ut.array_decode(ut.array_encode(s))
