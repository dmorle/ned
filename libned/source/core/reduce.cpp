#include <ned/core/reduce.h>

namespace nn
{
	namespace core
	{
		MdEdgeRef MdGraph::make_edge()
		{
			MdEdgeRef ret{ this, edges.size() };
			edges.push_back(MdEdge());
			return ret;
		}

		MdNodeRef MdGraph::make_node()
		{
			MdNodeRef ret{ this, nodes.size() };
			nodes.push_back(MdNode());
			return ret;
		}

		bool MdGraph::init(Graph& graph)
		{

		}

		bool MdGraph::alloc_edge(const Edge& edge)
		{
			sizeof(MdEdgeRef);
		}
	}
}
