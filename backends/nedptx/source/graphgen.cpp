#include <nedptx/graphgen.h>

namespace npx
{
	bool OpLoader::translate(nn::core::MdGraph* pgraph, nn::core::MdNodeRef node)
	{

	}

	GraphCompiler::GraphCompiler() :
		ctx(), mod("mod", ctx), builder(ctx), op_loader() {}

	bool GraphCompiler::init(nn::core::MdGraph& graph)
	{
		pgraph = &graph;

		// Running reduction passes over the graph
		for (auto& reduction_pass : reduction_passes)
			if (pgraph->run_pass(reduction_pass))
				return true;

		// I really don't want to have the backend do another graph translation.
		// we already have one from the builder* graph components to the core structure
		// and one for mode collapse.  So the nessisary info is just gonna get attached
		// to the MdGraph via the opaque pointer
		for (const auto& [name, edge] : pgraph->outs)
			if (init_edge(edge))
				return true;
		for (const auto& [name, edge] : pgraph->exps)
			if (init_edge(edge))
				return true;

		// inputs, outputs, and exports are all static
		for (const auto& [name, edge] : pgraph->inps)
			if (graph.opaque(edge))
				edge_data[graph.opaque<size_t>(edge)].is_static = true;
		for (const auto& [name, edge] : pgraph->outs)
			if (graph.opaque(edge))
				edge_data[graph.opaque<size_t>(edge)].is_static = true;
		for (const auto& [name, edge] : graph.exps)
			if (graph.opaque(edge))
				edge_data[graph.opaque<size_t>(edge)].is_static = true;

		return true;
	}

	bool GraphCompiler::compile_simple()
	{


		// Determining a memory requirement and memory offset for every edge in the graph
		size_t mem_size = 0;
		for (EdgeData& data : edge_data)
		{
			data.mem_offset = mem_size;
			mem_size += data.mem_required;
		}

		// Creating the step function which runs the entire model
		llvm::FunctionType* func_ty = llvm::FunctionType::get(llvm::Type::getVoidTy(ctx), false);
		llvm::Function* step_fn = llvm::Function::Create(func_ty,
			llvm::GlobalValue::LinkageTypes::ExternalLinkage, "step", mod);
		step_fn->setDLLStorageClass(llvm::GlobalValue::DLLExportStorageClass);  // __declspec(dllexport)
		llvm::BasicBlock* entry = llvm::BasicBlock::Create(mod.getContext(), "entry", step_fn);
		builder.SetInsertPoint(entry);



		// doing a DFS through the graph, compiling each of the nodes to a ptx kernel
		std::unordered_map<std::string, size_t> op_name_count;
		

		builder.CreateRetVoid();
	}

	bool GraphCompiler::init_edge(nn::core::MdEdgeRef edge)
	{
		// checking if the edge has already been initialized
		if (pgraph->opaque(edge))
			return false;

		// initializing the edge, opaque pointer is just used as an index into edge_data
		pgraph->opaque<size_t>(edge) = edge_data.size();
		edge_data.push_back({});

		// computing the minimum required memory to store the particular edge in bytes
		edge_data.back().mem_required = nn::core::fty_size(pgraph->get(edge).fp);
		for (size_t dim : pgraph->get(edge).shape)
			edge_data.back().mem_required *= dim;

		// TODO: figure out EdgeData

		// recursing on the edge input
		return init_node(pgraph->get(edge).inp.ref);
	}

	bool GraphCompiler::init_node(nn::core::MdNodeRef node)
	{
		// checking if the node has already been initialized
		if (pgraph->opaque(node))
			return false;

		// doing node translation
		if (op_loader.translate(pgraph, node))
			return true;

		// recursing on the node inputs
		for (const auto& [edge, view] : pgraph->get(node).inps)
			if (init_edge(edge))
				return true;
		return false;
	}

	bool GraphCompiler::compile_simple_node(llvm::Function* step,
		nn::core::MdNodeRef& node, std::unordered_map<std::string, size_t>& op_name_count)
	{
		llvm::FunctionType* func_ty = llvm::FunctionType::get(llvm::Type::getVoidTy(ctx), false);
		const std::string& base_name = pgraph->opaque<Op*>(node)->name;
		size_t idx = 0;
		if (op_name_count.contains(base_name))
			op_name_count[base_name] = idx;
		else
			idx = ++op_name_count.at(base_name);
		llvm::Function* kernel = llvm::Function::Create(func_ty,
			llvm::GlobalValue::LinkageTypes::InternalLinkage, base_name + std::to_string(idx), mod);
		LLVMInitializeNVPTXTarget();

		//llvm::BasicBlock* entry = llvm::BasicBlock::Create(ctx, "entry", func);
	}
}
