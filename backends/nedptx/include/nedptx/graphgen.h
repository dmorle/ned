#ifndef NPTX_GRAPHGEN_H
#define NPTX_GRAPHGEN_H

#include <nedptx/graphops.h>

#include <cuda_runtime.h>

namespace npx
{
	struct EdgeData
	{
		bool is_static;
		size_t mem_required;
		size_t mem_offset;
	};

	class OpLoader
	{
	public:
		OpLoader();

		// TODO: figure out an extensible method for defining and loading operations

		// automatically initializes the opaque pointer of the node
		bool translate(nn::core::MdGraph* graph, nn::core::MdNodeRef node);
	};

	class GraphCompiler
	{
		friend struct EdgeRef;
		friend struct NodeRef;

	public:
		GraphCompiler();

		bool init(nn::core::MdGraph& graph);

		// compiles the graph into a dynamic library where all the edges are
		// pre-allocated and the graph isn't optimized for the specific hardware
		// nodes are run synchronously by doing a DFS over the graph.
		bool compile_simple();

		bool load_reduction_passes();

	private:
		bool init_edge(nn::core::MdEdgeRef edge);
		bool init_node(nn::core::MdNodeRef node);

		bool compile_simple_node(llvm::Function* step,
			nn::core::MdNodeRef& node, std::unordered_map<std::string, size_t>& op_name_count);

		llvm::LLVMContext ctx;
		llvm::Module mod;
		llvm::IRBuilder<> builder;
		nn::core::MdGraph* pgraph = nullptr;

		// TODO: figure out a system for generically loading reduction passes
		std::vector<std::vector<nn::core::RedOp>> reduction_passes;

		OpLoader op_loader;
		std::vector<EdgeData> edge_data;
	};
}

#endif
