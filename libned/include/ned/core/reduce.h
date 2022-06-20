#ifndef NED_CORE_REDUCE_H
#define NED_CORE_REDUCE_H

#include <ned/core/config.h>
#include <ned/core/graph.h>

#include <vector>
#include <map>
#include <functional>

namespace nn
{
	namespace core
	{
		struct MdEdge;
		struct MdNode;
		class MdGraph;

		class MdEdgeRef
		{
			friend class MdGraph;

			MdEdgeRef(MdGraph* graph, size_t ptr) : graph(graph), ptr(ptr) {}
		public:
			MdEdgeRef(MdGraph* graph) : graph(graph), ptr(0) {}
			MdEdgeRef(const MdEdgeRef&) = default;
			MdEdgeRef(MdEdgeRef&&) = default;
			MdEdgeRef& operator=(const MdEdgeRef&) = default;
			MdEdgeRef& operator=(MdEdgeRef&&) = default;

			operator bool() const noexcept;
			MdEdge* operator->() noexcept;
			const MdEdge* operator->() const noexcept;
			MdEdge& operator*() noexcept;
			const MdEdge& operator*() const noexcept;

		private:
			MdGraph* graph;
			size_t ptr;
		};

		class MdNodeRef
		{
			friend class MdGraph;

			MdNodeRef(MdGraph* graph, size_t ptr) : graph(graph), ptr(ptr) {}
		public:
			MdNodeRef(MdGraph* graph) : graph(graph), ptr(0) {}
			MdNodeRef(const MdNodeRef&) = default;
			MdNodeRef(MdNodeRef&&) = default;
			MdNodeRef& operator=(const MdNodeRef&) = default;
			MdNodeRef& operator=(MdNodeRef&&) = default;

			operator bool() const noexcept;
			MdNode* operator->() noexcept;
			const MdNode* operator->() const noexcept;
			MdNode& operator*() noexcept;
			const MdNode& operator*() const noexcept;

		private:
			MdGraph* graph;
			size_t ptr;
		};

		struct EdgeView
		{
			MdEdgeRef edge;
			size_t elem_offset;  // The element offset in memory of the edge view
			size_t nelems;  // The number of elements in the edge view

			std::vector<size_t> shape;
			std::vector<size_t> strides;

			static EdgeView identity(MdEdge* pedge);
		};

		struct MdEdge
		{
			// describing the edge
			EdgeFty fp;
			std::vector<size_t> shape;

			// the edge's connections
			MdNode* inp;
			std::vector<MdNode*> outs;

			// fields used for memory layout
			size_t mem_offset;
		};

		struct MdNode
		{
			std::string name;
			std::map<std::string, ConfigVal> configs;

			std::vector<EdgeView> inps;
			std::vector<EdgeView> outs;
		};

		struct MdTensor
		{
			MdEdgeRef forward;
			MdEdgeRef backward;
		};

		struct MdParameter
		{
			MdEdgeRef forward;
			MdEdgeRef backward;
			void* data = nullptr;
		};

		class MdGraph
		{
			friend class MdEdgeRef;
			friend class MdNodeRef;
			using RedOp = std::function<void(MdGraph&, MdNodeRef)>;

		public:
			MdGraph() {}

			// Generates the MdGraph and does automatic pruning
			bool init(Graph& graph);

			// Does a pass over the graph applying each of the reductions as it goes
			bool run_pass(const std::vector<MdGraph&, RedOp>& ops);

			MdEdgeRef make_edge();
			MdNodeRef make_node();

		private:
			// Used by init for mapping the opaque pointers in the graph's nodes and edges
			bool alloc_edge(const Edge& edge);
			bool alloc_node(const Node& node);  

			// memory management stuff
			std::vector<MdEdge> edges = { MdEdge() };
			std::vector<MdNode> nodes = { MdNode() };

			// References to edges in the graph
			std::map<std::string, MdTensor> inps;
			std::map<std::string, MdTensor> outs;
			std::map<std::string, MdTensor> exps;
			std::map<std::string, MdParameter> exts;
		};
	}
}

#endif
