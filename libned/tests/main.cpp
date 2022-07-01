#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <fstream>

#include <ned/errors.h>
#include <ned/lang/bytecode.h>
#include <ned/lang/interp.h>
#include <ned/lang/compiler.h>
#include <ned/core/graph.h>
#include <ned/core/reduce.h>

using namespace nn;
using namespace lang;
using namespace core;

// too big to fit on the actual stack
CallStack stack;

int main(void)
{
    std::cout << "print test" << std::endl;

    if (false) {
        TokenArray tarr;
        if (lex_file(TESTS_DIR"sum.bcnn", tarr))
        {
            error::print();
            return 1;
        }

        ProgramHeap heap;
        ByteCodeModule mod{ heap };
        if (parsebc_module(tarr, mod))
        {
            error::print();
            return 1;
        }
        std::cout << mod.to_string();

        ByteCode byte_code;
        GraphBuilder builder;
        Obj obj, fwd, bwd;

        if (mod.export_module(byte_code) ||
            // creating the cargs
            heap.create_obj_fty(obj, EdgeFty::F32) ||
            stack.push(obj) ||
            heap.create_obj_int(obj, 10) ||
            stack.push(obj) ||
            // creating the vargs
            builder.create_edg(fwd, EdgeInfo{ .fty = EdgeFty::F32, .dims = {10} }) ||
            builder.create_edg(bwd, EdgeInfo{ .fty = EdgeFty::F32, .dims = {10} }) ||
            builder.create_tsr(obj) ||
            builder.set_fwd(obj.ptr, fwd.ptr) ||
            builder.set_bwd(obj.ptr, bwd.ptr) ||
            stack.push(obj) ||  // lhs
            builder.create_edg(fwd, EdgeInfo{ .fty = EdgeFty::F32, .dims = {10} }) ||
            builder.create_edg(bwd, EdgeInfo{ .fty = EdgeFty::F32, .dims = {10} }) ||
            builder.create_tsr(obj) ||
            builder.set_fwd(obj.ptr, fwd.ptr) ||
            builder.set_bwd(obj.ptr, bwd.ptr) ||
            stack.push(obj) ||  // rhs
            stack.push(Obj{ .ptr = 0 }) ||  // null block, this will mark the first block as root
            exec(stack, heap, builder, byte_code, "model"))
        {
            error::print();
            return 1;
        }

        core::Edge* edge = new core::Edge();

        delete edge;

        core::Graph graph;
        if (builder.export_graph(graph))
        {
            error::print();
            return 1;
        }
    }
    
    if (true) {
        TokenArray tarr;
        if (lex_file(TESTS_DIR"adding.nn", tarr))
        {
            error::print();
            return 0;
        }

        AstModule ast;
        if (parse_module(tarr, ast))
        {
            error::print();
            return 0;
        }

        ProgramHeap heap;
        ByteCodeModule bc{ heap };
        ModuleInfo info;
        if (codegen_module(bc, info, ast, {}))
        {
            error::print();
            return 0;
        }

        TypeManager manager{};
        info.init(&manager);
        TypeRef N = info.create_int(5);
        TypeRef M = info.create_int(6);
        if (!N || !M)
        {
            error::print();
            return 0;
        }
        TypeRef fp = info.create_fty(core::EdgeFty::F16);
        TypeRef shape = info.create_array({ N, M });
        if (!fp || !shape || info.entry_setup("model", { {"fp", fp}, {"shape", shape} }))
        {
            error::print();
            return 0;
        }

        std::ofstream ofs{ TESTS_DIR"bcdump/adding.bcnn" };
        ofs << bc.to_string() << std::endl;

        ByteCode byte_code;
        GraphBuilder builder;
        if (bc.export_module(byte_code) || exec(stack, heap, builder, byte_code, "main"))
        {
            error::print();
            return 0;
        }

        core::Graph graph;
        if (builder.export_graph(graph))
        {
            error::print();
            return 0;
        }

        // Attaching an SGD optimizer
        std::vector<GraphMod> opt;
        for (const auto& [name, param] : graph.weights)
        {
            // replacing proc main with a new one for the optimizer
            bc.del_block("main");
            TypeManager manager{};
            info.init(&manager);
            assert(param.forward->info == param.backward->info);
            
            // For testing, I'm just gonna assume that info.create_* doesn't fail
            TypeRef lr = info.create_float(1e-3);
            TypeRef fp = info.create_fty(param.forward->info.fty);
            std::vector<TypeRef> elems;
            for (size_t dim : param.forward->info.dims)
                elems.push_back(info.create_int(dim));
            TypeRef shape = info.create_array(elems);
            if (info.entry_setup("SGD", { { "lr", lr }, {"fp", fp}, {"shape", shape}}))
            {
                error::print();
                return 0;
            }

            std::ofstream ofs{ TESTS_DIR"bcdump/adding_sgd.bcnn" };
            ofs << bc.to_string() << std::endl;
            
            ByteCode byte_code;
            GraphBuilder sgd_builder;
            if (bc.export_module(byte_code) || exec(stack, heap, sgd_builder, byte_code, "main"))
            {
                error::print();
                return 0;
            }

            core::Graph sgd;
            if (sgd_builder.export_graph(sgd))
            {
                error::print();
                return 0;
            }

            opt.push_back({{
                .inp_map = {{{InpRef::Type::WEIGHT, name}, {OutRef::Type::OUTPUT, "weight"}}},
                .out_map = {}
            }, sgd});
        }

        if (attach_graph(graph, "training", opt))
        {
            error::print();
            return 0;
        }

        core::MdGraph md_graph{ "" };
        if (md_graph.init(graph))
        {
            error::print();
            return 0;
        }

        return 0;
    }

    return 0;
}
