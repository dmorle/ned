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

		struct EdgeView
		{
			struct Axis
			{
				size_t stride;
				size_t start;
				size_t end;
				// applied on top of stride
				// negative values means to index from the end to the start
				int step;
			};

			std::vector<Axis> axes;
		};

		struct MdNodeRef { size_t val = 0; operator bool() { return val; } };
		struct MdEdgeRef { size_t val = 0; operator bool() { return val; } };

		struct MdEdgeConnector { MdEdgeRef ref; EdgeView view; };
		struct MdNodeConnector { MdNodeRef ref; size_t idx; };

		struct MdEdge
		{
			// describing the edge
			EdgeFty fp = EdgeFty::F32;
			std::vector<size_t> shape;

			// the edge's connections
			MdNodeConnector inp = { 0 };
			std::vector<MdNodeConnector> outs;
			
			// The weight info if its the forward edge of a model weight.  Otherwise null
			void* data = nullptr;

			mutable void* opaque;
		};

		EdgeView identity_edge_view(const MdEdge& edge);

		struct MdNode
		{
			std::string name;
			std::map<std::string, Config> configs;

			std::vector<MdEdgeConnector> inps;
			std::vector<MdEdgeConnector> outs;

			mutable void* opaque;
		};

		struct MdParameter
		{
			MdEdgeRef forward;
			MdEdgeRef backward;
			void* data = nullptr;
		};

		using RedOp = std::function<void(MdGraph&, MdNodeRef)>;

		class MdGraph
		{
			friend struct MdEdgeRef;
			friend struct MdNodeRef;

		public:
			MdGraph(const std::string& exec_md) : exec_md(exec_md) {}

			// Generates the MdGraph and does automatic pruning
			bool init(Graph& graph);

			// Does a pass over the graph applying each of the reductions as it goes
			bool run_pass(const std::vector<RedOp>& ops);

			MdEdgeRef make_edge();
			MdNodeRef make_node();

			MdEdge& get(MdEdgeRef ref) noexcept;
			const MdEdge& get(MdEdgeRef ref) const noexcept;
			MdNode& get(MdNodeRef ref) noexcept;
			const MdNode& get(MdNodeRef ref) const noexcept;

			// Helper function for retrieving the opaque field of an MdEdge object and
			// casting it to the type provided in the template argument.
			// By default it gets cast to void*
			template<typename T = void*>
			T& opaque(MdEdgeRef ref) noexcept
			{
				static_assert(sizeof(T) == 8);
				return *reinterpret_cast<T*>(&edges[ref.val].opaque);
			}
			
			// Helper function for retrieving the opaque field of an MdNode object and
			// casting it to the type provided in the template argument.
			// By default, it gets cast to void*
			template<typename T = void*>
			T& opaque(MdNodeRef ref) noexcept
			{
				static_assert(sizeof(T) == 8);
				return *reinterpret_cast<T*>(&nodes[ref.val].opaque);
			}

		private:
			// Only going backwards through the graph to initialize all the edges and nodes
			bool init_edge(const Edge& edge);
			bool init_node(const Node& node);

			// DFS through the graph again, this time binding the outputs of edges and nodes
			bool bind_edge(const Edge& edge);
			bool bind_node(const Node& node);

			bool check_mode(const std::string& mode);

			void* as_ptr(MdEdgeRef ref) const noexcept;
			void* as_ptr(MdNodeRef ref) const noexcept;
			MdEdgeRef edge_ptr(void* ptr) const noexcept;
			MdNodeRef node_ptr(void* ptr) const noexcept;

			std::string exec_md;

			// memory management stuff
			std::vector<MdEdge> edges = { MdEdge() };
			std::vector<MdNode> nodes = { MdNode() };

		public:
			// References to edges in the graph
			std::map<std::string, MdEdgeRef> inps;
			std::map<std::string, MdEdgeRef> outs;
			std::map<std::string, MdEdgeRef> exps;
			std::map<std::string, MdEdgeRef> exts;
		};
	}
}

#endif
