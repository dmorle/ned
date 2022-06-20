#include <ned/core/reduce.h>

namespace nn
{
	namespace core
	{
		MdEdgeRef::operator bool() const noexcept { return ptr; }
		MdEdge* MdEdgeRef::operator->() noexcept { return &graph->edges[ptr]; }
		const MdEdge* MdEdgeRef::operator->() const noexcept { return &graph->edges[ptr]; }
		MdEdge& MdEdgeRef::operator*() noexcept { return  graph->edges[ptr]; }
		const MdEdge& MdEdgeRef::operator*() const noexcept { return  graph->edges[ptr]; }

		MdNodeRef::operator bool() const noexcept { return ptr; }
		MdNode* MdNodeRef::operator->() noexcept { return &graph->nodes[ptr]; }
		const MdNode* MdNodeRef::operator->() const noexcept { return &graph->nodes[ptr]; }
		MdNode& MdNodeRef::operator*() noexcept { return  graph->nodes[ptr]; }
		const MdNode& MdNodeRef::operator*() const noexcept { return  graph->nodes[ptr]; }

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
			return true;
		}

		bool MdGraph::alloc_edge(const Edge& edge)
		{
			return true;
		}
	}
}
