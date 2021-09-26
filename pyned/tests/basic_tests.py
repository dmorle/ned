import unittest

from pyned import lang
from pyned import core


class TestSGE(unittest.TestCase):
    """
    Single Graph Edge tests
    """

    def assert_single_graph_edge(self, edge, dty):
        self.assertEqual(edge.dtype, dty)
        self.assertIsNone(edge.input)
        self.assertEqual(len(edge.outputs), 0)

    def test_decl_scalar(self):
        ast = lang.parse_file("tests/nn/basic_tests/decl_scalar.nn")
        graph = ast.eval("model")

        # Graph inputs
        self.assertEqual(len(graph.inputs), 1)
        self.assertEqual(graph.inputs.keys()[0], "model-out-0")
        self.assert_single_graph_edge(graph.inputs["model-out-0"], core.dtype.f32)
        self.assertEqual(graph.inputs["model-out-0"].shape, [])

        # Graph outputs
        self.assertEqual(len(graph.outputs), 1)
        self.assert_single_graph_edge(graph.outputs[0], core.dtype.f32)
        self.assertEqual(graph.outputs[0].shape, [])

    def test_inp_scalar(self):
        ast = lang.parse_file("tests/nn/basic_tests/inp_scalar.nn")
        graph = ast.eval("model")

        # Graph inputs
        self.assertEqual(len(graph.inputs), 1)
        self.assertEqual(graph.inputs.keys()[0], "inp")
        self.assert_single_graph_edge(graph.inputs["inp"], core.dtype.f32)
        self.assertEqual(graph.inputs["inp"].shape, [])

        # Graph outputs
        self.assertEqual(len(graph.outputs), 1)
        self.assert_single_graph_edge(graph.outputs[0], core.dtype.f32)
        self.assertEqual(graph.outputs[0].shape, [])

    def test_decl_tensor_5(self):
        shape = [5]
        ast = lang.parse_file("tests/nn/basic_tests/decl_tensor.nn")
        graph = ast.eval("model", shape)
        
        # Graph inputs
        self.assertEqual(len(graph.inputs), 1)
        self.assertEqual(graph.inputs.keys()[0], "model-out-0")
        self.assert_single_graph_edge(graph.inputs["model-out-0"], core.dtype.f32)
        self.assertEqual(graph.inputs["model-out-0"].shape, shape)

        # Graph outputs
        self.assertEqual(len(graph.outputs), 1)
        self.assert_single_graph_edge(graph.outputs[0], core.dtype.f32)
        self.assertEqual(graph.outputs[0].shape, shape)

    def test_decl_tensor_5_5_5(self):
        shape = [5, 5, 5]
        ast = lang.parse_file("tests/nn/basic_tests/decl_tensor.nn")
        graph = ast.eval("model", shape)

        # Graph inputs
        self.assertEqual(len(graph.inputs), 1)
        self.assertEqual(graph.inputs.keys()[0], "model-out-0")
        self.assert_single_graph_edge(graph.inputs["model-out-0"], core.dtype.f32)
        self.assertEqual(graph.inputs["model-out-0"].shape, shape)

        # Graph outputs
        self.assertEqual(len(graph.outputs), 1)
        self.assert_single_graph_edge(graph.outputs[0], core.dtype.f32)
        self.assertEqual(graph.outputs[0].shape, shape)

    def test_inp_tensor_5(self):
        shape = [5]
        ast = lang.parse_file("tests/nn/basic_tests/inp_tensor.nn")
        graph = ast.eval("model", shape)
        
        # Graph inputs
        self.assertEqual(len(graph.inputs), 1)
        self.assertEqual(graph.inputs.keys()[0], "inp")
        self.assert_single_graph_edge(graph.inputs["inp"], core.dtype.f32)
        self.assertEqual(graph.inputs["inp"].shape, shape)

        # Graph outputs
        self.assertEqual(len(graph.outputs), 1)
        self.assert_single_graph_edge(graph.outputs[0], core.dtype.f32)
        self.assertEqual(graph.outputs[0].shape, shape)

    def test_inp_tensor_5_5_5(self):
        shape = [5, 5, 5]
        ast = lang.parse_file("tests/nn/basic_tests/inp_tensor.nn")
        graph = ast.eval("model", shape)
        
        # Graph inputs
        self.assertEqual(len(graph.inputs), 1)
        self.assertEqual(graph.inputs.keys()[0], "inp")
        self.assert_single_graph_edge(graph.inputs["inp"], core.dtype.f32)
        self.assertEqual(graph.inputs["inp"].shape, shape)

        # Graph outputs
        self.assertEqual(len(graph.outputs), 1)
        self.assert_single_graph_edge(graph.outputs[0], core.dtype.f32)
        self.assertEqual(graph.outputs[0].shape, shape)


class TestUnaryNode(unittest.TestCase):
    def assert_identity_node(self, inp, out):
        pass

    def test_decl_unary_node(self):
        dty = core.dtype.f32
        shape = [5, 5, 5]
        ast = lang.parse_file("tests/nn/basic_tests/decl_unarynode.nn")
        graph = ast.eval("model", [dty] + shape)

        self.assertEqual(len(graph.inputs), 1)
        self.assertEqual(graph.inputs.keys()[0], "model-weight-0")
        self.assertEqual(graph.inputs["model-weight-0"].dtype, dty)

        self.assertEqual(len(graph.outputs), 1)
        self.assertEqual(graph.outputs[0].dtype, dty)

        # TODO: make sure the node was created properly


if __name__ == "__main__":
    unittest.main()
