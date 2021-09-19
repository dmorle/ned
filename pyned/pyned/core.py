from enum import Enum
from typing import List, Optional, Iterable

from _pyned import core as _core


class dtype(Enum):
    f16 = "f16"
    f32 = "f32"
    f64 = "f64"


class Edge: pass
class Node: pass


class NodeInputs:
    def __init__(self, node: _core.Node):
        assert type(node) is _core.Node
        self._node = node

    def __len__(self) -> int:
        return self._node.input_size()
    
    def __getitem__(self, idx) -> Edge:
        assert type(idx) is int
        return Edge(self._node.get_input(idx))
        
    def __iter__(self) -> Iterable[Edge]:
        return iter([self._node.get_input(i) for i in range(len(self))])


class NodeOutputs:
    def __init__(self, node: _core.Node):
        assert type(node) is _core.Node
        self._node = node

    def __len__(self) -> int:
        return self._node.output_size()
    
    def __getitem__(self, idx) -> Edge:
        assert type(idx) is int
        return Edge(self._node.get_output(idx))
        
    def __iter__(self) -> Iterable[Edge]:
        return iter([self._node.get_output(i) for i in range(len(self))])


class Node:
    def __init__(self, node: _core.Node):
        assert type(node) is _core.Node
        assert node.is_valid()
        self._node = node

        from .lang import Obj
        self.Obj = Obj

    @property
    def name(self) -> str:
        return self._node.get_name()

    @property
    def inputs(self) -> NodeInputs:
        return NodeInputs(self._node)

    @property
    def outputs(self) -> NodeOutputs:
        return NodeOutputs(self._node)

    def has_opaque(self) -> bool:
        return self._node.has_opaque()

    def get_cargs(self):
        return [self.Obj(e) for e in self._node.get_cargs()]


class EdgeOutputs:
    def __init__(self, edge: _core.Edge):
        assert type(edge) is _core.Edge
        assert edge.is_valid()
        self._edge = edge

    def __len__(self) -> int:
        return self._edge.output_size()

    def __getitem__(self, idx: int) -> Node:
        return self._edge.get_output(idx)

    def __iter__(self) -> Iterable[Edge]:
        return iter([self._edge.get_output(i) for i in range(len(self))])


class Edge:
    def __init__(self, edge: _core.Edge):
        assert type(edge) is _core.Edge
        assert edge.is_valid()
        self._edge = edge

    @property
    def dtype(self) -> dtype:
        dty = self._edge.get_fwidth()
        if dty == 0:
            return dtype.f16
        elif dty == 1:
            return dtype.f32
        elif dty == 2:
            return dtype.f64
        raise ValueError("Unable to determine data type for graph edge")

    @property
    def shape(self) -> List[int]:
        return self._edge.get_shape()

    @property
    def input(self) -> Optional[Node]:
        if self._edge.has_input():
            return self._edge.get_input()[0]
        return None

    @property
    def input_id(self) -> Optional[int]:
        if self._edge.has_input():
            return self._edge.get_input()[1]
        return None

    @property
    def outputs(self) -> EdgeOutputs:
        return EdgeOutputs(self._edge)

    @property
    def output_ids(self) -> List[int]:
        sz = self._edge.output_size()
        ids = list()
        for i in range(sz):
            ids.append(self._edge.get_output(i)[1])
        return ids


class GraphInputs:
    def __init__(self, graph):
        assert type(graph) is _core.Graph
        self._graph = graph

    def __len__(self) -> int:
        return self._graph.input_size()

    def __getitem__(self, key: str) -> Edge:
        assert type(key) is str
        return Edge(self._graph.get_input(key))

    def __iter__(self) -> Iterable[str]:
        return iter(self.keys())

    def keys(self) -> List[str]:
        return self._graph.list_inputs()


class GraphOutputs:
    def __init__(self, graph):
        assert type(graph) is _core.Graph
        self._graph = graph

    def __len__(self) -> int:
        return self._graph.output_size()
    
    def __getitem__(self, idx) -> Edge:
        assert type(idx) is int
        return Edge(self._graph.get_output(idx))

    def __iter__(self) -> Iterable[Edge]:
        return iter([self._graph.get_output(i) for i in range(len(self))])


class Graph:
    def __init__(self, graph):
        assert type(graph) is _core.Graph
        assert graph.is_valid()
        self._graph = graph

    @property
    def inputs(self) -> GraphInputs:
        return GraphInputs(self._graph)

    @property
    def outputs(self) -> GraphOutputs:
        return GraphOutputs(self._graph)

