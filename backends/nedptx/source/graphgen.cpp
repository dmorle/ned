#include <nedptx/graphgen.h>

namespace npx
{
	bool OpLoader::translate(nn::core::MdGraph& graph, nn::core::MdNodeRef node)
	{

	}

	GraphCompiler::GraphCompiler() :
		ctx(), mod("mod", ctx), builder(ctx), op_loader() {}

	bool GraphCompiler::init(nn::core::MdGraph& graph)
	{
		// Running reduction passes over the graph
		for (auto& reduction_pass : reduction_passes)
			if (graph.run_pass(reduction_pass))
				return true;

		// I really don't want to have the backend to another graph translation.
		// we already have one from the builder* graph components to the core structure
		// and one for mode collapse.  So the nessisary info is just gonna get attached
		// to the MdGraph via the opaque pointer
		for (const auto& [name, edge] : graph.outs)
			if (init_edge(graph, edge))
				return true;

		return true;
	}

	bool GraphCompiler::init_edge(nn::core::MdGraph& graph, nn::core::MdEdgeRef edge)
	{
		// checking if the edge has already been initialized
		if (graph.opaque(edge))
			return false;

		// initializing the edge, opaque pointer is just used as an index into edge_data
		graph.opaque<size_t>(edge) = edge_data.size();
		edge_data.push_back({});

		// computing the minimum required memory to store the particular edge in bytes
		edge_data.back().mem_required = nn::core::fty_size(graph.get(edge).fp);
		for (size_t dim : graph.get(edge).shape)
			edge_data.back().mem_required *= dim;

		// TODO: figure out EdgeData

		// recursing on the edge input
		return init_node(graph, graph.get(edge).inp.ref);
	}

	bool GraphCompiler::init_node(nn::core::MdGraph& graph, nn::core::MdNodeRef node)
	{
		// checking if the node has already been initialized
		if (graph.opaque(node))
			return false;

		// doing node translation
		if (op_loader.translate(graph, node))
			return true;

		// recursing on the node inputs
		for (const auto& [edge, view] : graph.get(node).inps)
			if (init_edge(graph, edge))
				return true;
		return false;
	}
}
