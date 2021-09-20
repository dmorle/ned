import unittest

from pyned import lang
from pyned.core import dtype


class TestSGE(unittest.TestCase):
    def assert_single_graph_edge(self, edge, dty):
        self.assertEqual(edge.dtype, dty)
        self.assertIsNone(edge.input)
        self.assertEqual(len(edge.outputs), 0)

    def test_decl_scalar(self):
        ast = lang.parse_file("tests/nn/decl_scalar.nn")
        graph = ast.eval("model")

        # Graph inputs
        self.assertEqual(len(graph.inputs), 1)
        self.assertEqual(graph.inputs.keys()[0], "model-out-0")
        self.assert_single_graph_edge(graph.inputs["model-out-0"], dtype.f32)
        self.assertEqual(graph.inputs["model-out-0"].shape, [])

        # Graph outputs
        self.assertEqual(len(graph.outputs), 1)
        self.assert_single_graph_edge(graph.outputs[0], dtype.f32)
        self.assertEqual(graph.outputs[0].shape, [])

    def test_inp_scalar(self):
        ast = lang.parse_file("tests/nn/inp_scalar.nn")
        graph = ast.eval("model")

        # Graph inputs
        self.assertEqual(len(graph.inputs), 1)
        self.assertEqual(graph.inputs.keys()[0], "inp")
        self.assert_single_graph_edge(graph.inputs["inp"], dtype.f32)
        self.assertEqual(graph.inputs["inp"].shape, [])

        # Graph outputs
        self.assertEqual(len(graph.outputs), 1)
        self.assert_single_graph_edge(graph.outputs[0], dtype.f32)
        self.assertEqual(graph.outputs[0].shape, [])

    def test_decl_tensor_5(self):
        ast = lang.parse_file("tests/nn/decl_tensor.nn")
        graph = ast.eval("model", lang.Obj(5))
        
        # Graph inputs
        self.assertEqual(len(graph.inputs), 1)
        self.assertEqual(graph.inputs.keys()[0], "model-out-0")
        self.assert_single_graph_edge(graph.inputs["model-out-0"], dtype.f32)
        self.assertEqual(graph.inputs["model-out-0"].shape, [5])

        # Graph outputs
        self.assertEqual(len(graph.outputs), 1)
        self.assert_single_graph_edge(graph.outputs[0], dtype.f32)
        self.assertEqual(graph.outputs[0].shape, [5])

    def test_decl_tensor_5_5_5(self):
        ast = lang.parse_file("tests/nn/decl_tensor.nn")
        graph = ast.eval("model", lang.Obj(5), lang.Obj(5), lang.Obj(5))

        # Graph inputs
        self.assertEqual(len(graph.inputs), 1)
        self.assertEqual(graph.inputs.keys()[0], "model-out-0")
        self.assert_single_graph_edge(graph.inputs["model-out-0"], dtype.f32)
        self.assertEqual(graph.inputs["model-out-0"].shape, [5, 5, 5])

        # Graph outputs
        self.assertEqual(len(graph.outputs), 1)
        self.assert_single_graph_edge(graph.outputs[0], dtype.f32)
        self.assertEqual(graph.outputs[0].shape, [5, 5, 5])

    def test_inp_tensor_5(self):
        ast = lang.parse_file("tests/nn/inp_tensor.nn")
        graph = ast.eval("model", lang.Obj(5))
        
        # Graph inputs
        self.assertEqual(len(graph.inputs), 1)
        self.assertEqual(graph.inputs.keys()[0], "inp")
        self.assert_single_graph_edge(graph.inputs["inp"], dtype.f32)
        self.assertEqual(graph.inputs["inp"].shape, [5])

        # Graph outputs
        self.assertEqual(len(graph.outputs), 1)
        self.assert_single_graph_edge(graph.outputs[0], dtype.f32)
        self.assertEqual(graph.outputs[0].shape, [5])

    def test_inp_tensor_5_5_5(self):
        ast = lang.parse_file("tests/nn/inp_tensor.nn")
        graph = ast.eval("model", lang.Obj(5), lang.Obj(5), lang.Obj(5))
        
        # Graph inputs
        self.assertEqual(len(graph.inputs), 1)
        self.assertEqual(graph.inputs.keys()[0], "inp")
        self.assert_single_graph_edge(graph.inputs["inp"], dtype.f32)
        self.assertEqual(graph.inputs["inp"].shape, [5, 5, 5])

        # Graph outputs
        self.assertEqual(len(graph.outputs), 1)
        self.assert_single_graph_edge(graph.outputs[0], dtype.f32)
        self.assertEqual(graph.outputs[0].shape, [5, 5, 5])


if __name__ == "__main__":
    unittest.main()
