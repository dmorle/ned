#include <nvm/graphgen.h>

#include <iostream>
#include <fstream>
#include <cassert>

#ifdef WIN32
#include <windows.h>
#include <locale>
#include <codecvt>
#endif

namespace nvm
{

    void mark_inp_edge(EdgeData& edge_data, const std::string& name)
    {
        edge_data.is_static = true;
        edge_data.static_data = { StaticData::Type::INPUT, name };
    }

    void mark_out_edge(EdgeData& edge_data, const std::string& name)
    {
        edge_data.is_static = true;
        edge_data.static_data = { StaticData::Type::OUTPUT, name };
    }

    void mark_exp_edge(EdgeData& edge_data, const std::string& name)
    {
        edge_data.is_static = true;
        edge_data.static_data = { StaticData::Type::EXPORT, name };
    }

    void mark_ext_edge(EdgeData& edge_data, const std::string& name)
    {
        edge_data.is_static = true;
        edge_data.static_data = { StaticData::Type::EXTERN, name };
    }

    DepInfo DepInfo::from_static_data(const StaticData& static_data)
    {
        switch (static_data.ty)
        {
        case StaticData::Type::INPUT:
            return { DepInfo::Type::INPUT, static_data.name };
        case StaticData::Type::OUTPUT:
            return { DepInfo::Type::OUTPUT, static_data.name };
        case StaticData::Type::EXPORT:
            return { DepInfo::Type::EXPORT, static_data.name };
        case StaticData::Type::EXTERN:
            return { DepInfo::Type::EXTERN, static_data.name };
        default:
            assert(false);
            return {};
        }
    }

    GraphCompiler::GraphCompiler() {}

    bool GraphCompiler::init(nn::core::MdGraph& graph)
    {
        pgraph = &graph;

        // TODO: run reduction passes over the graph

        // Initializing the opaque pointers for each of the edges and nodes
        for (const auto& [name, edge] : graph.outs)
            if (init_edge(edge))
                return true;
        for (const auto& [name, edge] : graph.exps)
            if (init_edge(edge))
                return true;

        // inputs, outputs, externs, and exports are all static
        for (const auto& [name, edge] : graph.inps)
            if (graph.opaque(edge))
                mark_inp_edge(edge_data[graph.opaque<size_t>(edge)], name);
        for (const auto& [name, edge] : graph.outs)
            if (graph.opaque(edge))
                mark_out_edge(edge_data[graph.opaque<size_t>(edge)], name);
        for (const auto& [name, edge] : graph.exps)
            if (graph.opaque(edge))
                mark_exp_edge(edge_data[graph.opaque<size_t>(edge)], name);
        for (const auto& [name, edge] : graph.exts)
            if (graph.opaque(edge))
                mark_ext_edge(edge_data[graph.opaque<size_t>(edge)], name);

        return false;
    }

    bool GraphCompiler::init_edge(nn::core::MdEdgeRef edge)
    {
        // checking if the edge has already been initialized
        if (pgraph->opaque(edge))
            return false;

        // initializing the edge, opaque pointer is just used as an index into edge_data
        pgraph->opaque<size_t>(edge) = edge_data.size();
        edge_data.push_back({});

        return init_node(pgraph->get(edge).inp.ref);
    }

    bool GraphCompiler::init_node(nn::core::MdNodeRef node)
    {
        pgraph->opaque<size_t>(node) = 0;  // index into sync_names
        for (const auto& [edge, view] : pgraph->get(node).inps)
        {
            if (init_edge(edge))
                return true;
            edge_data[pgraph->opaque<size_t>(edge)].n_locks++;  // locking the feeding node
        }
        return false;
    }

    void GraphCompiler::print()
    {
        mod.print(llvm::outs(), nullptr);
    }

    bool GraphCompiler::compile()
    {
        using namespace nn;

        // Initializing the helper functions needed for compilation
        streq = compile_streq();
        memcpy = compile_memcpy();

        // Generating the edges ahead of time since I need them to all be ready
        // before compiling any nodes
        for (core::MdEdgeRef edge_ref : pgraph->list_edges())
            compile_edge(edge_ref);

        // Looking through each of the nodes in the graph for any sync nodes
        std::map<std::string, std::vector<core::MdNodeRef>> sync_map;
        for (core::MdNodeRef node_ref : pgraph->list_nodes())
        {
            const auto& node = pgraph->get(node_ref);
            if (node.name != sync_node_name)
                continue;

            // Making sure the sync node's signature is correct
            if (assert_sync_node_sig(node))
                return true;
            assert(node.configs.at("name").ty == core::Config::Tag::STRING);
            sync_map[node.configs.at("name").val_str].push_back(node_ref);
        }

        // Compiling each of the syncs
        std::map<std::string, std::vector<DepInfo>> sync_deps;
        std::map<std::string, llvm::Function*> sync_funcs;
        for (const auto& [name, node_refs] : sync_map)
        {
            // Generating a sync id for the current sync
            size_t sync_id = sync_names.size();
            sync_names.push_back(name);

            // Compiling each of the nodes needed for the currrent sync.
            // Along the way, adding the node functions to the funcs vector
            // before returing from each function in the dfs, which automatically gives me a
            // valid execution order
            std::vector<llvm::Function*> funcs;
            for (core::MdNodeRef node_ref : node_refs)
                if (compile_sync_node(node_ref, funcs, sync_deps[name], sync_id))
                    return true;
            
            // Generating the code needed to run the current sync.
            // The sync node itself will be implemented by a memcpy
            sync_funcs[name] = compile_sync_fn(std::string("run_sync_") + name, funcs, node_refs);
        }

        // Generating the run_sync function which takes in the name of a sync as a c-string
        // and calls the underlying run_sync_* function
        compile_run_sync(sync_funcs);

        // Generating a special "sync" for computing the graph outputs and exports.
        // It's not really a sync since it's span on the graph isn't determined by sync nodes,
        // and it can't be called from run_sync function - it doesn't even have a name.
        // Instead, it will be run by calling the "run" function, which will
        // always guarenteed to be there by the abi.
        std::vector<core::MdNodeRef> run_node_refs;
        std::vector<DepInfo> run_deps;
        for (const auto& [name, edge_ref] : pgraph->outs)
        {
            const core::MdEdge& edge = pgraph->get(edge_ref);
            if (!edge.inp.ref)  // Checking if the edge has a feeding node
                continue;
            
            const core::MdNode& node = pgraph->get(edge.inp.ref);
            if (node.name == "sync")  // Skipping sync nodes - just add them to the run dependancies
                run_deps.push_back({ DepInfo::Type::SYNC, node.configs.at("name").val_str });
            else
                run_node_refs.push_back(edge.inp.ref);
        }
        // Exports also get computed on a run call
        for (const auto& [name, edge_ref] : pgraph->exps)
        {
            const core::MdEdge& edge = pgraph->get(edge_ref);
            if (!edge.inp.ref)  // Checking if the edge has a feeding node
                continue;

            const core::MdNode& node = pgraph->get(edge.inp.ref);
            if (node.name == "sync")  // Skipping sync nodes - just add them to the run dependancies
                run_deps.push_back({ DepInfo::Type::SYNC, node.configs.at("name").val_str });
            else
                run_node_refs.push_back(edge.inp.ref);
        }

        // Compiling the nodes that the run function depends on.  And to do that I need to
        // add a pseudo-sync id for the compile_node function to check claims against
        size_t run_sync_id = sync_names.size();
        sync_names.push_back("~run~");
        std::vector<llvm::Function*> run_funcs;
        for (core::MdNodeRef node_ref : run_node_refs)
            if (compile_sync_node(node_ref, run_funcs, run_deps, run_sync_id))
                return true;

        // Compiling the run function itself
        llvm::Type* void_ty = llvm::Type::getVoidTy(ctx);
        llvm::FunctionType* run_fn_ty = llvm::FunctionType::get(void_ty, false);
        llvm::Function* run_fn = llvm::Function::Create(run_fn_ty,
            llvm::GlobalValue::LinkageTypes::ExternalLinkage, "run", mod);
        run_fn->setDLLStorageClass(llvm::GlobalValue::DLLStorageClassTypes::DLLExportStorageClass);
        llvm::BasicBlock* entry = llvm::BasicBlock::Create(ctx, "entry", run_fn);
        builder.SetInsertPoint(entry);
        for (llvm::Function* func : run_funcs)
            builder.CreateCall(func);
        builder.CreateRetVoid();

        // Generating the edge io functions
        compile_edge_io(pgraph->inps, "inp");
        compile_edge_io(pgraph->outs, "out");
        compile_edge_io(pgraph->exps, "exp");
        compile_edge_io(pgraph->exts, "ext");

        // TODO: figure out how to communicate sync dependancies over the abi

        // Compiling to a dll
        return compile_dll();
    }

    void GraphCompiler::compile_edge(nn::core::MdEdgeRef edge_ref)
    {
        EdgeData& data = edge_data[pgraph->opaque<size_t>(edge_ref)];
        static size_t edge_idx = 0;

        // computing the number of elements in the edge and building a static array from it
        uint64_t sz = 1;
        for (size_t dim : pgraph->get(edge_ref).shape)
            sz *= dim;
        auto edge_ty = llvm::ArrayType::get(get_fptype(ctx, pgraph->get(edge_ref).fp), sz);
        std::string edge_name = "edge_" + std::to_string(edge_idx++);
        data.var = nullptr;
        mod.getOrInsertGlobal(edge_name, edge_ty, [&] {
            data.var = new llvm::GlobalVariable(mod, edge_ty, false, llvm::GlobalVariable::InternalLinkage,
                llvm::ConstantAggregateZero::get(edge_ty), edge_name);
            return data.var; });
        assert(data.var);
    }

    bool GraphCompiler::compile_sync_node(
        nn::core::MdNodeRef node_ref,
        std::vector<llvm::Function*>& funcs,
        std::vector<DepInfo>& sync_deps,
        size_t sync_id)
    {
        using namespace nn;
        const core::MdNode& node = pgraph->get(node_ref);

        // Checking if the node has already been compiled
        size_t existing_sync_id = pgraph->opaque<size_t>(node_ref);
        if (existing_sync_id != 0)
        {
            // Making sure that no other sync nodes are claiming the current node
            if (existing_sync_id == sync_id)
                return false;

            return error::graph(
                "During compilation of node % for sync %,\n"
                "it was found that the node was already compiled for sync %.\n"
                "A node must belong to either a unique sync, or to no sync at all.",
                node.name, sync_names[sync_id], sync_names[existing_sync_id]);
        }

        // Compiling the feeding nodes before compiling the current node
        // unless that node is a sync node - those will be taken care of in the
        // corresponding run_sync_* function
        for (const auto& [edge_ref, edge_view] : node.inps)
        {
            const auto& edge_data = get_edge_data(edge_ref);
            if (edge_data.is_static)
            {
                // Stopping at any static nodes since for the output static edge types
                // There's already the "run" function which is used to construct them,
                // and for input static edge types, well they're inputs...
                sync_deps.push_back(DepInfo::from_static_data(edge_data.static_data));
                continue;
            }
            core::MdNodeRef inp_node_ref = pgraph->get(edge_ref).inp.ref;
            if (!inp_node_ref)  // Checking if the edge had any feeding node at all
                continue;
            const core::MdNode& inp_node = pgraph->get(inp_node_ref);
            if (inp_node.name == sync_node_name)
            {
                // Skipping sync nodes, but adding them as a dependancy of the current sync
                sync_deps.push_back({ DepInfo::Type::SYNC , inp_node.configs.at("name").val_str });
                continue;
            }
            if (compile_normal_node(inp_node_ref, funcs, sync_deps, sync_id))
                return true;
        }
        
        return false;
    }

    bool GraphCompiler::compile_normal_node(
        nn::core::MdNodeRef node_ref,
        std::vector<llvm::Function*>& funcs,
        std::vector<DepInfo>& sync_deps,
        size_t sync_id)
    {
        // Compiling a normal node is just like compiling a sync node,
        // except you actually need to generate code for running the computation
        if (compile_sync_node(node_ref, funcs, sync_deps, sync_id))
            return true;

        // Creating the llvm function
        const nn::core::MdNode& node = pgraph->get(node_ref);
        size_t idx = node_name_counts[node.name]++;
        llvm::FunctionType* func_ty = llvm::FunctionType::get(llvm::Type::getVoidTy(ctx), false);
        llvm::Function* node_fn = llvm::Function::Create(func_ty,
            llvm::GlobalValue::LinkageTypes::InternalLinkage, pgraph->get(node_ref).name + "_" + std::to_string(idx), mod);

        // Using the plugin to compile the node
        CompCtx comp_ctx = { this, &mod, &builder, node_fn };
        //if (compile_node())
        //    return nn::error::graph("Failed to compile node %", node.name);
        funcs.push_back(node_fn);
        pgraph->opaque<size_t>(node_ref) = sync_id;  // marking the node as having been compiled

        return false;
    }

    void GraphCompiler::compile_edge_io(const std::map<std::string, nn::core::MdEdgeRef>& edges, const std::string& name)
    {
        llvm::Type* i8_ty = llvm::Type::getInt8Ty(ctx);
        llvm::Type* i32_ty = llvm::Type::getInt32Ty(ctx);
        llvm::Type* i8_ptr_ty = i8_ty->getPointerTo();
        llvm::Type* string_ty = i8_ty->getPointerTo();
        llvm::Value* success_ret = llvm::ConstantInt::get(i32_ty, 0);
        llvm::Value* failed_ret = llvm::ConstantInt::get(i32_ty, 1);
        llvm::FunctionType* fn_ty = llvm::FunctionType::get(i32_ty, { { string_ty, i8_ptr_ty } }, false);

        // The getter and setter functions are very similar. The only difference is the direction of the memcpy
        auto compile_func = [&](
            const std::string& func_name,
            const std::function<void(llvm::Value*, nn::core::MdEdgeRef)>& handle_match) -> void
        {
            llvm::Function* fn = llvm::Function::Create(fn_ty,
                llvm::GlobalValue::LinkageTypes::ExternalLinkage, func_name, mod);
            fn->setDLLStorageClass(llvm::GlobalValue::DLLStorageClassTypes::DLLExportStorageClass);
            llvm::Argument* given_edge_name = fn->getArg(0);
            llvm::Argument* given_ptr = fn->getArg(1);
            llvm::BasicBlock* entry = llvm::BasicBlock::Create(ctx, "entry", fn);
            builder.SetInsertPoint(entry);

            for (const auto& [edge_name, edge_ref] : edges)
            {
                // Setting up the blocks needed for branching on the string comparison
                llvm::BasicBlock* true_branch = llvm::BasicBlock::Create(ctx, std::string("its_") + edge_name, fn);
                llvm::BasicBlock* false_branch = llvm::BasicBlock::Create(ctx, std::string("not_") + edge_name, fn);

                // Creating a global constant that contains edge_name as a c-string
                std::string global_name = get_unique_global_name(name + "_" + edge_name);
                auto [name_type, name_initializer] = get_litstr(edge_name);
                llvm::GlobalVariable* name_var = nullptr;
                mod.getOrInsertGlobal(global_name, name_type, [&] {
                    name_var = new llvm::GlobalVariable(mod, name_type, false, llvm::GlobalVariable::InternalLinkage,
                        name_initializer, edge_name);
                    return name_var; });
                assert(name_var);

                // Doing the string comparison
                llvm::Value* match_result = builder.CreateCall(streq, { given_edge_name, name_var });
                builder.CreateCondBr(match_result, true_branch, false_branch);

                // If it matched, memcpy and return 0
                builder.SetInsertPoint(true_branch);
                handle_match(given_ptr, edge_ref);
                builder.CreateRet(success_ret);

                // If it missed, continue on checking
                builder.SetInsertPoint(false_branch);
            }
            builder.CreateRet(failed_ret);
        };

        // Getter function
        compile_func(
            std::string("get_") + name,
            [&](llvm::Value* given_ptr, nn::core::MdEdgeRef edge_ref) -> void {
                const nn::core::MdEdge& edge = pgraph->get(edge_ref);
                const EdgeData& data = edge_data[pgraph->opaque<size_t>(edge_ref)];
                size_t nbytes = nn::core::fty_size(edge.fp);
                for (size_t dim : edge.shape)
                    nbytes *= dim;
                auto nbytes_val = llvm::ConstantInt::get(i32_ty, nbytes);
                auto data_var = builder.CreateCast(llvm::Instruction::CastOps::BitCast, data.var, i8_ptr_ty);
                builder.CreateCall(memcpy, { given_ptr, data_var, nbytes_val });
            });

        // Setter function
        compile_func(
            std::string("set_") + name,
            [&](llvm::Value* given_ptr, nn::core::MdEdgeRef edge_ref) -> void {
                const nn::core::MdEdge& edge = pgraph->get(edge_ref);
                const EdgeData& data = edge_data[pgraph->opaque<size_t>(edge_ref)];
                size_t nbytes = nn::core::fty_size(edge.fp);
                for (size_t dim : edge.shape)
                    nbytes *= dim;
                auto nbytes_val = llvm::ConstantInt::get(i32_ty, nbytes);
                auto data_var = builder.CreateCast(llvm::Instruction::CastOps::BitCast, data.var, i8_ptr_ty);
                builder.CreateCall(memcpy, { data_var, given_ptr, nbytes_val });
            });
    }

    void GraphCompiler::compile_run_sync(const std::map<std::string, llvm::Function*>& sync_funcs)
    {
        llvm::Type* i8_ty = llvm::Type::getInt8Ty(ctx);
        llvm::Type* i32_ty = llvm::Type::getInt32Ty(ctx);
        llvm::Type* string_ty = i8_ty->getPointerTo();
        llvm::Value* success_ret = llvm::ConstantInt::get(i32_ty, 0);
        llvm::Value* failed_ret = llvm::ConstantInt::get(i32_ty, 1);
        llvm::FunctionType* run_sync_fn_ty = llvm::FunctionType::get(i32_ty, { { string_ty } }, false);
        llvm::Function* run_sync_fn = llvm::Function::Create(run_sync_fn_ty,
            llvm::GlobalValue::LinkageTypes::ExternalLinkage, "run_sync", mod);
        run_sync_fn->setDLLStorageClass(llvm::GlobalValue::DLLStorageClassTypes::DLLExportStorageClass);
        llvm::Argument* given_sync_name = run_sync_fn->getArg(0);
        llvm::BasicBlock* entry = llvm::BasicBlock::Create(ctx, "entry", run_sync_fn);
        builder.SetInsertPoint(entry);

        for (const auto& [sync_name, sync_fn] : sync_funcs)
        {
            // Setting up the blocks needed for branching on the string comparison
            llvm::BasicBlock* true_branch = llvm::BasicBlock::Create(ctx, std::string("its_") + sync_name, run_sync_fn);
            llvm::BasicBlock* false_branch = llvm::BasicBlock::Create(ctx, std::string("not_") + sync_name, run_sync_fn);

            // Creating a global constant that contains edge_name as a c-string
            std::string global_name = get_unique_global_name(std::string("run_sync_") + sync_name);
            auto [name_type, name_initializer] = get_litstr(sync_name);
            llvm::GlobalVariable* name_var = nullptr;
            mod.getOrInsertGlobal(global_name, name_type, [&] {
                name_var = new llvm::GlobalVariable(mod, name_type, false, llvm::GlobalVariable::InternalLinkage,
                    name_initializer, sync_name);
                return name_var; });
            assert(name_var);

            // Doing the string comparison
            llvm::Value* match_result = builder.CreateCall(streq, { given_sync_name, name_var });
            builder.CreateCondBr(match_result, true_branch, false_branch);

            // If it matched, memcpy and return 0
            builder.SetInsertPoint(true_branch);
            builder.CreateCall(sync_fn);
            builder.CreateRet(success_ret);

            // If it missed, continue on checking
            builder.SetInsertPoint(false_branch);
        }
        builder.CreateRet(failed_ret);
    }

    llvm::Function* GraphCompiler::compile_sync_fn(
        const std::string& name,
        const std::vector<llvm::Function*>& funcs,
        const std::vector<nn::core::MdNodeRef>& node_refs)
    {
        using namespace nn;

        llvm::Type* void_ty = llvm::Type::getVoidTy(ctx);
        llvm::FunctionType* sync_fn_ty = llvm::FunctionType::get(void_ty, false);
        llvm::Function* sync_fn = llvm::Function::Create(sync_fn_ty,
            llvm::GlobalValue::LinkageTypes::InternalLinkage, name, mod);
        
        llvm::Type* i32_ty = llvm::Type::getInt32Ty(ctx);
        llvm::BasicBlock* entry = llvm::BasicBlock::Create(ctx, "entry", sync_fn);
        builder.SetInsertPoint(entry);
        for (llvm::Function* func : funcs)
            builder.CreateCall(func);

        // Doing all the memcpys at once
        for (core::MdNodeRef node_ref : node_refs)
        {
            const core::MdNode& node = pgraph->get(node_ref);
            assert(node.name == sync_node_name);
            size_t nbytes = core::fty_size(node.configs.at("fp").val_fty);
            for (auto& e : node.configs.at("shape").val_list)
                nbytes *= e.val_int;
            llvm::Value* nbytes_val = llvm::ConstantInt::get(i32_ty, nbytes);
            auto& [inp_edge_ref, inp_edge_view] = node.inps[0];
            auto& [out_edge_ref, out_edge_view] = node.outs[0];
            // TODO: handle edge views
            EdgeData& inp_edge_data = get_edge_data(inp_edge_ref);
            EdgeData& out_edge_data = get_edge_data(out_edge_ref);
            builder.CreateCall(memcpy, { out_edge_data.var, inp_edge_data.var, nbytes_val });
        }

        builder.CreateRetVoid();
        return sync_fn;
    }

    bool GraphCompiler::compile_dll()
    {
#ifdef WIN32
        // Windows is weird
        llvm::Type* ty = llvm::Type::getInt32Ty(ctx);
        mod.getOrInsertGlobal("_fltused", ty, [&]() -> llvm::GlobalVariable* {
            return new llvm::GlobalVariable(mod, ty, true, llvm::GlobalVariable::ExternalLinkage,
                llvm::ConstantInt::get(ty, 0), "_fltused"); });
#endif
        
        {
            std::error_code ec;
            llvm::raw_fd_ostream ofs("test.bc", ec);
            // TODO: check ec
            llvm::WriteBitcodeToFile(mod, ofs);
        }

        {
            std::error_code ec;
            llvm::raw_fd_ostream ofs("test.ll", ec);
            mod.print(ofs, nullptr);
        }

#ifdef WIN32
        // This is extremely hacky, but nothing else worked
        wchar_t buf[512];
        GetCurrentDirectory(512, buf);
        std::string curr_dir = std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t>().to_bytes(buf);
        if (system("llc -filetype=obj test.bc") ||
            system("lld-link /dll /noentry test.obj")
            ) return true;
#endif
        return false;
    }

    bool GraphCompiler::assert_sync_node_sig(const nn::core::MdNode& node)
    {
        using namespace nn;
        decltype(node.configs)::const_iterator it;

        // Checking the cargs

        it = node.configs.find("name");
        if (it == node.configs.end())
            return error::graph("Found a sync node without a name carg");
        if (it->second.ty != core::Config::Tag::STRING)
            return error::graph("Found a sync node with a non-string name carg");

        it = node.configs.find("fp");
        if (it == node.configs.end())
            return error::graph("Found a sync node without an fp carg");
        if (it->second.ty != core::Config::Tag::FTY)
            return error::graph("Found a sync node with a non-fty fp carg");

        it = node.configs.find("shape");
        if (it == node.configs.end())
            return error::graph("Found a sync node without a shape carg");
        if (it->second.ty != core::Config::Tag::LIST)
            return error::graph("Found a sync node with a non-array<int> shape carg");
        for (const auto& elem : it->second.val_list)
            if (elem.ty != core::Config::Tag::INT)
                return error::graph("Found a sync node with a non-array<int> shape carg");

        if (node.configs.size() != 3)
        {
            std::vector<std::string> unexpected_cargs;
            for (const auto& [carg_name, cfg] : node.configs)
                if (carg_name != "name" &&
                    carg_name != "fp" &&
                    carg_name != "shape"
                    ) unexpected_cargs.push_back(carg_name);
            return error::graph("Found a sync node with unexpected cargs: %", unexpected_cargs);
        }

        // Checking the vargs

        if (node.inps.size() != 1)
            return error::graph("Found a sync node with % vargs, expected 1", node.inps.size());
        if (node.outs.size() != 1)
            return error::graph("Found a sync node with % rets, expected 1", node.outs.size());

        return false;
    }

    llvm::Function* GraphCompiler::compile_memcpy()
    {
        // This is very inefficient implementation, but I just want to get something working for now ...
        llvm::Type* i8_ty = llvm::Type::getInt8Ty(ctx);
        llvm::Type* i8_ptr_ty = i8_ty->getPointerTo();
        llvm::Type* i32_ty = llvm::Type::getInt32Ty(ctx);
        llvm::Type* void_ty = llvm::Type::getVoidTy(ctx);
        llvm::FunctionType* memcpy_fn_ty = llvm::FunctionType::get(void_ty, { {i8_ptr_ty, i8_ptr_ty, i32_ty} }, false);
        llvm::Function* memcpy_fn = llvm::Function::Create(memcpy_fn_ty,
            llvm::GlobalValue::LinkageTypes::InternalLinkage, "nvm_memcpy", mod);
        llvm::Argument* dst = memcpy_fn->getArg(0);
        llvm::Argument* src = memcpy_fn->getArg(1);
        llvm::Argument* nbytes = memcpy_fn->getArg(2);

        llvm::Value* step_val = llvm::ConstantInt::get(i32_ty, 1);
        llvm::Value* i32z_val = llvm::ConstantInt::get(i32_ty, 0);

        llvm::BasicBlock* entry = llvm::BasicBlock::Create(ctx, "entry", memcpy_fn);
        llvm::BasicBlock* loop  = llvm::BasicBlock::Create(ctx, "loop",  memcpy_fn);
        llvm::BasicBlock* end   = llvm::BasicBlock::Create(ctx, "end",   memcpy_fn);

        builder.SetInsertPoint(entry);                                 // entry:
        builder.CreateBr(loop);                                        //     br label %loop
        
        builder.SetInsertPoint(loop);                                  // loop:
        auto idx  = builder.CreatePHI(i32_ty, 2, "idx");               //     %idx = phi i32 [0, %entry], [%nidx, %loop]
        auto pdst = builder.CreateGEP(i8_ty, dst, { {idx} }, "pdst");  //     %pdst = getelementptr i8, i8* %dst, i32 %idx
        auto psrc = builder.CreateGEP(i8_ty, src, { {idx} }, "psrc");  //     %psrc = getelementptr i8, i8* %src, i32 %idx
        auto tmp  = builder.CreateLoad(i8_ty, psrc, "tmp");            //     %tmp = load i8, i8* %psrc
                    builder.CreateStore(tmp, pdst);                    //     store i8 %tmp, i8* %pdst
        auto nidx = builder.CreateAdd(idx, step_val, "nidx");          //     %nidx = add i32 %idx, 1
        auto cond = builder.CreateICmpEQ(nidx, nbytes, "cond");        //     %cond = icmp eq i32 %nidx, %nbytes
        builder.CreateCondBr(cond, end, loop);                         //     br i1 %cond, label %end, label %loop

        builder.SetInsertPoint(end);                                   // end:
        builder.CreateRetVoid();                                       //     ret void

        idx->addIncoming(i32z_val, entry);
        idx->addIncoming(nidx, loop);

        return memcpy_fn;
    }

    llvm::Function* GraphCompiler::compile_streq()
    {
        llvm::Type* i1_ty = llvm::Type::getInt1Ty(ctx);
        llvm::Type* i8_ty = llvm::Type::getInt8Ty(ctx);
        llvm::Type* i8_ptr_ty = i8_ty->getPointerTo();
        llvm::Type* i32_ty = llvm::Type::getInt32Ty(ctx);
        llvm::FunctionType* streq_fn_ty = llvm::FunctionType::get(i1_ty, { {i8_ptr_ty, i8_ptr_ty} }, false);
        llvm::Function* streq_fn = llvm::Function::Create(streq_fn_ty,
            llvm::GlobalValue::LinkageTypes::InternalLinkage, "nvm_streq", mod);

        llvm::Argument* lhs = streq_fn->getArg(0);
        llvm::Argument* rhs = streq_fn->getArg(1);

        llvm::Value*  step_val = llvm::ConstantInt::get(i32_ty, 1);
        llvm::Value*  i32z_val = llvm::ConstantInt::get(i32_ty, 0);
        llvm::Value*   i8z_val = llvm::ConstantInt::get(i8_ty,  0);
        llvm::Value*  true_val = llvm::ConstantInt::get(i1_ty,  1);
        llvm::Value* false_val = llvm::ConstantInt::get(i1_ty,  0);

        llvm::BasicBlock* entry     = llvm::BasicBlock::Create(ctx, "entry",     streq_fn);
        llvm::BasicBlock* loop      = llvm::BasicBlock::Create(ctx, "loop",      streq_fn);
        llvm::BasicBlock* eq        = llvm::BasicBlock::Create(ctx, "eq",        streq_fn);
        llvm::BasicBlock* cont      = llvm::BasicBlock::Create(ctx, "cont",      streq_fn);
        llvm::BasicBlock* ret_true  = llvm::BasicBlock::Create(ctx, "ret_true",  streq_fn);
        llvm::BasicBlock* ret_false = llvm::BasicBlock::Create(ctx, "ret_false", streq_fn);
        
        builder.SetInsertPoint(entry);                                     // entry:
        builder.CreateBr(loop);                                            //     br label %loop

        builder.SetInsertPoint(loop);                                      // loop:
        auto idx    = builder.CreatePHI(i32_ty, 2, "idx");                 //     %idx = phi i32 [0, %entry], [%next_idx, %cont]
        auto lchptr = builder.CreateGEP(i8_ty, lhs, { {idx} }, "lchptr");  //     %lchptr = getelementptr i8, i8* %lhs, i32 %idx
        auto rchptr = builder.CreateGEP(i8_ty, rhs, { {idx} }, "rchptr");  //     %rchptr = getelementptr i8, i8* %rhs, i32 %idx
        auto lch    = builder.CreateLoad(i8_ty, lchptr, "lch");            //     %lch = load i8, i8* %lchptr
        auto rch    = builder.CreateLoad(i8_ty, rchptr, "rch");            //     %rch = load i8, i8* %rchptr
        auto cheq   = builder.CreateICmpNE(lch, rch, "cheq");              //     %cheq = icmp ne i8 %lch, %rch
        builder.CreateCondBr(cheq, ret_false, eq);                         //     br i1 %cheq, label %ret_false, label %eq

        builder.SetInsertPoint(eq);                                        // eq:
        auto chz = builder.CreateICmpEQ(lch, i8z_val, "chz");              //     %chz = icmp eq i8 %lch, 0
        builder.CreateCondBr(chz, ret_true, cont);                         //     br i1 %chz, label %ret_true, label %cont

        builder.SetInsertPoint(cont);                                      // cont:
        auto next_idx = builder.CreateAdd(idx, step_val, "next_idx");      //     %next_idx = add i32 %idx, 1
        builder.CreateBr(loop);                                            //     br label %loop

        builder.SetInsertPoint(ret_true);                                  // ret_true:
        builder.CreateRet(true_val);                                       //     ret i1 1

        builder.SetInsertPoint(ret_false);                                 // ret_false:
        builder.CreateRet(false_val);                                      //     ret i1 0

        idx->addIncoming(i32z_val, entry);
        idx->addIncoming(next_idx, cont);

        return streq_fn;
    }

    std::pair<llvm::ArrayType*, llvm::Constant*> GraphCompiler::get_litstr(const std::string& str)
    {
        llvm::Type* i8_ty = llvm::Type::getInt8Ty(ctx);
        llvm::ArrayType* str_type = llvm::ArrayType::get(i8_ty, str.size() + 1);
        std::vector<llvm::Constant*> vals;
        vals.reserve(str.size() + 1);
        for (char c : str)
            vals.push_back(llvm::ConstantInt::get(i8_ty, c));
        vals.push_back(llvm::ConstantInt::get(i8_ty, 0));
        llvm::Constant* str_value = llvm::ConstantArray::get(str_type, vals);
        return { str_type, str_value };
    }

    std::string GraphCompiler::get_unique_global_name(const std::string& prefix)
    {
        size_t idx = 0;
        while (mod.getGlobalVariable(prefix + std::to_string(idx))) idx += 1;
        return prefix + std::to_string(idx);
    }
    
    EdgeData& GraphCompiler::get_edge_data(nn::core::MdEdgeRef edge)
    { return edge_data[pgraph->opaque<size_t>(edge)]; }

    const EdgeData& GraphCompiler::inp_edge(const nn::core::MdNode& node, size_t idx) const
    { return edge_data[pgraph->opaque<size_t>(node.inps[idx].ref)]; }

    const EdgeData& GraphCompiler::out_edge(const nn::core::MdNode& node, size_t idx) const
    { return edge_data[pgraph->opaque<size_t>(node.outs[idx].ref)]; }

}
