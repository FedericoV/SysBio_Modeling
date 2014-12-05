__author__ = 'Federico Vaggi'


from ..project.utils import OrderedHashDict
from unittest import TestCase
from nose.tools import raises


class TestProject(TestCase):
    @raises(TypeError)
    def test_hashdict_rejects_ints(self):
        hashdict = OrderedHashDict()
        hashdict[5] = 'hello'

    @raises(TypeError)
    def test_hashdict_rejects_tuples(self):
        hashdict = OrderedHashDict()
        hashdict[('hello', 'hi')] = 'hello'

    def test_hashdict_normal_lookup(self):
        hashdict = OrderedHashDict()
        hashdict['hi'] = 'hello'
        hashdict['a'] = 'a'
        assert (hashdict['a'] == 'a')

    def test_hashdict_set_lookup(self):
        hashdict = OrderedHashDict()
        hashdict['hi'] = 'hello'
        hashdict[frozenset(['a', 'b'])] = 'set'
        assert (hashdict['a'] == 'set')
        assert (hashdict['b'] == 'set')

    def test_hashdict_membership(self):
        hashdict = OrderedHashDict()
        hashdict[frozenset(['a', 'b'])] = 'set'
        assert ('b' in hashdict)
        assert ('a' in hashdict)


    @raises(KeyError)
    def test_hashdict_items_present_in_set(self):
        hashdict = OrderedHashDict()
        hashdict[frozenset(['a', 'b'])] = 'hi'
        hashdict[frozenset(['c', 'b'])] = 'set'

    @raises(KeyError)
    def test_hashdict_items_present_in_set2(self):
        hashdict = OrderedHashDict()
        hashdict[frozenset(['a', 'b'])] = 'hi'
        hashdict['a'] = 'set'

    @raises(KeyError)
    def test_hashdict_items_present_in_set3(self):
        hashdict = OrderedHashDict()
        hashdict[frozenset(['a'])] = 'hi'
        hashdict['a'] = 'set'

    @raises(KeyError)
    def test_hashdict_items_present_in_set4(self):
        hashdict = OrderedHashDict()
        hashdict['aafdadf'] = 'set'
        hashdict[frozenset(['aafdadf', 'b'])] = 'hi'

    @raises(TypeError)
    def test_nonstring_frozenset(self):
        hashdict = OrderedHashDict()
        hashdict[frozenset(['hi', 3])] = 'hello'

    def test_hashdict_partial_duplicates(self):
        hashdict = OrderedHashDict()
        hashdict['a'] = 'set'
        hashdict[frozenset(['ab', 'b'])] = 'hi'

        assert (hashdict['a'] == 'set')
        assert (hashdict['ab'] == 'hi')
        assert (hashdict['b'] == 'hi')

    def test_hashdict_itersequence(self):
        import random
        nums = range(200)
        random.shuffle(nums)
        chars = [chr(n) for n in nums]
        hashdict = OrderedHashDict()

        for k, v in zip(chars, nums):
            hashdict[k] = v

        assert (chars == hashdict.keys())

    def test_overwrite_value(self):
        hashdict = OrderedHashDict()
        hashdict['Total'] = 5
        hashdict['Total'] = 6

        assert (hashdict['Total'] == 6)

