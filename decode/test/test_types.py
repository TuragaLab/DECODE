from copy import deepcopy
from decode import utils


class TestRecursiveNamespace:
    d = {'a': 1,
         'b': {
             'c': 2,
             'd': 3
            }
         }

    dr = utils.types.RecursiveNamespace()
    dr.a = 1
    dr.b = utils.types.RecursiveNamespace()
    dr.b.c = 2
    dr.b.d = 3

    def test_dict2namespace(self):

        """Run"""
        dr = utils.types.RecursiveNamespace(**self.d)

        """Assertions"""
        assert dr.a == 1
        assert dr.b.c == 2
        assert dr.b.d == 3

    def test_namespace2dict(self):

        """Run"""
        d = self.dr.to_dict()

        """Assertions"""
        assert d['a'] == 1
        assert d['b']['c'] == 2
        assert d['b']['d'] == 3

    def test_cycle(self):

        """Run and Assert"""
        assert utils.types.RecursiveNamespace(**self.d).to_dict() == self.d
        assert utils.types.RecursiveNamespace(**self.dr.to_dict()) == self.dr
