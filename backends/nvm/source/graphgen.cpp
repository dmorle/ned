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

    GraphCompiler::GraphCompiler() {}

    GraphCompiler::~GraphCompiler()
    {
        using namespace nn::util;
        for (Library* lib : plugin_libs)
            lib_del(lib);  // Nothing I can do if this fails
    }

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
                edge_data[graph.opaque<size_t>(edge)].is_static = true;
        for (const auto& [name, edge] : graph.outs)
            if (graph.opaque(edge))
                edge_data[graph.opaque<size_t>(edge)].is_static = true;
        for (const auto& [name, edge] : graph.exps)
            if (graph.opaque(edge))
                edge_data[graph.opaque<size_t>(edge)].is_static = true;
        for (const auto& [name, edge] : graph.exts)
            if (graph.opaque(edge))
                edge_data[graph.opaque<size_t>(edge)].is_static = true;

        // finding the plugins needed for compilation
        find_plugins(node_map, plugin_libs);

        return false;
    }

    void GraphCompiler::print()
    {
        mod.print(llvm::outs(), nullptr);
    }

    bool GraphCompiler::compile()
    {
        // Initializing the helper functions needed for compilation
        streq = compile_streq();
        memcpy = compile_memcpy();

        // Generating the code for each of the nodes and allocating the edges
        // Along the way, adding the node functions to the funcs vector
        // before returing from each function in the dfs, which automatically gives me a
        // valid execution order
        std::vector<llvm::Function*> funcs;
        for (const auto& [name, edge] : pgraph->outs)
            if (pgraph->opaque(edge) && compile_edge(edge, funcs))
                return true;
        for (const auto& [name, edge] : pgraph->exps)
            if (pgraph->opaque(edge) && compile_edge(edge, funcs))
                return true;
        
        // Generating the step function
        llvm::Type* void_ty = llvm::Type::getVoidTy(ctx);
        llvm::FunctionType* step_fn_ty = llvm::FunctionType::get(void_ty, false);
        llvm::Function* step_fn = llvm::Function::Create(step_fn_ty,
            llvm::GlobalValue::LinkageTypes::ExternalLinkage, "step", mod);
        step_fn->setDLLStorageClass(llvm::GlobalValue::DLLStorageClassTypes::DLLExportStorageClass);
        llvm::BasicBlock* entry = llvm::BasicBlock::Create(ctx, "entry", step_fn);
        builder.SetInsertPoint(entry);
        for (llvm::Function* func : funcs)
            builder.CreateCall(func);
        builder.CreateRetVoid();

        // Generating the edge io functions
        compile_edge_io(pgraph->inps, "inp");
        compile_edge_io(pgraph->outs, "out");

        // Compilation to a dll
        return compile_dll();
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
        // checking if the node has already been initialized
        if (pgraph->opaque(node))
            return false;

        // TODO: Figure out NodeData

        for (const auto& [edge, view] : pgraph->get(node).inps)
        {
            if (init_edge(edge))
                return true;
            edge_data[pgraph->opaque<size_t>(edge)].n_locks++;  // locking the feeding node
        }
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

    bool GraphCompiler::compile_edge(nn::core::MdEdgeRef edge_ref, std::vector<llvm::Function*>& funcs)
    {
        // Checking if it has already been initialized
        if (!pgraph->opaque(edge_ref))
            return false;

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

        // Compiling the edge's input - if it exists
        if (!pgraph->get(edge_ref).inp.ref)
            return false;
        return compile_node(pgraph->get(edge_ref).inp.ref, funcs);
    }

    bool GraphCompiler::compile_node(nn::core::MdNodeRef node_ref, std::vector<llvm::Function*>& funcs)
    {
        const auto& node = pgraph->get(node_ref);

        // Compiling the input edges before compiling the node
        for (const auto& [edge_ref, edge_view] : node.inps)
            if (compile_edge(edge_ref, funcs))
                return true;

        // Finding the appropriate overload for the node
        if (!node_map.contains(node.name))
            return nn::error::graph("Unable to find any overloads for node name %", node.name);
        NodeCtx node_ctx = { node_ref, pgraph };
        const NodeImpl* node_match = nullptr;
        for (const auto& node_impl : node_map.at(node.name))
            if (node_impl.match(node_ctx))
            {
                node_match = &node_impl;
                break;
            }
        if (!node_match)
            return nn::error::graph("Unable to find a matching over load for node %", node.name);

        // Creating the llvm function
        size_t idx = node_name_counts[node.name]++;
        llvm::FunctionType* func_ty = llvm::FunctionType::get(llvm::Type::getVoidTy(ctx), false);
        llvm::Function* node_fn = llvm::Function::Create(func_ty,
            llvm::GlobalValue::LinkageTypes::InternalLinkage, pgraph->get(node_ref).name + "_" + std::to_string(idx), mod);

        // Using the plugin to compile the node
        CompCtx comp_ctx = { this, &mod, &builder, node_fn };
        if (node_match->compile(node_ctx, comp_ctx))
            return nn::error::graph("Failed to compile node %", node.name);
        funcs.push_back(node_fn);

        return false;
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
