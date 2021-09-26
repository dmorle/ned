import unittest

from pyned import lang
from pyned import core


class TestSame(unittest.TestCase):
    def test_decl_same(self):
        dty = core.dtype.f32
        shape = [5, 5, 5]
        ast = lang.parse_file("tests/nn/binop_tests/decl_same.nn")
        graph = ast.eval("model", [dty] + shape)

        # TODO: implement the test

    def test_inp_same(self):
        dty = core.dtype.f32
        shape = [5, 5, 5]
        ast = lang.parse_file("tests/nn/binop_tests/inp_same.nn")
        graph = ast.eval("model", [dty] + shape)

        # TODO: implement the test


if __name__ == "__main__":
    unittest.main()
