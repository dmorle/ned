#ifndef NPTX_GRAPHGEN_H
#define NPTX_GRAPHGEN_H

#include <nedptx/graphops.h>

#include <cuda_runtime.h>

namespace npx
{
	struct EdgeData
	{
		size_t mem_required;
	};

	class OpLoader
	{
	public:
		OpLoader();

		// TODO: figure out an extensible method for defining and loading operations

		// automatically initializes the opaque pointer of the node
		bool translate(nn::core::MdGraph& graph, nn::core::MdNodeRef node);
	};

	class GraphCompiler
	{
		friend struct EdgeRef;
		friend struct NodeRef;

	public:
		GraphCompiler();

		bool init(nn::core::MdGraph& graph);

		bool load_reduction_passes();

	private:
		bool init_edge(nn::core::MdGraph& graph, nn::core::MdEdgeRef edge);
		bool init_node(nn::core::MdGraph& graph, nn::core::MdNodeRef node);

		llvm::LLVMContext ctx;
		llvm::Module mod;
		llvm::IRBuilder<> builder;

		// TODO: figure out a system for generically loading reduction passes
		std::vector<std::vector<nn::core::RedOp>> reduction_passes;

		OpLoader op_loader;
		std::vector<EdgeData> edge_data;
	};
}

#endif
