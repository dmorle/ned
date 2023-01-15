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

bool generate_graph(const std::string& nnfname, const std::string& bcfname, core::Graph& graph, std::function<bool(ModuleInfo&)> setup)
{
    TokenArray tarr;
    std::string fpth = std::string(TESTS_DIR) + nnfname;
    if (lex_file(fpth.c_str(), tarr))
        return true;

    AstModule ast;
    if (parse_module(tarr, ast))
        return true;

    ProgramHeap heap;
    ByteCodeModule bc{ heap };
    ModuleInfo info;
    if (codegen_module(bc, info, ast, {}))
        return true;

    TypeManager manager{};
    info.init(&manager);
    if (setup(info))
        return true;

    if (bcfname.size() != 0)
    {
        std::ofstream ofs{ std::string(TESTS_DIR) + "bcdump/" + bcfname };
        ofs << bc.to_string() << std::endl;
    }

    ByteCode byte_code;
    GraphBuilder builder;
    return
        bc.export_module(byte_code) ||
        exec(stack, heap, builder, byte_code, "main") ||
        builder.export_graph(graph);
}

bool test_sum()
{
    TokenArray tarr;
    if (lex_file(TESTS_DIR"sum.bcnn", tarr))
        return true;

    ProgramHeap heap;
    ByteCodeModule mod{ heap };
    if (parsebc_module(tarr, mod))
        return true;
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
        exec(stack, heap, builder, byte_code, "model")
        ) return true;

    core::Graph graph;
    return builder.export_graph(graph);
}

bool test_adding()
{
    core::Graph graph;
    auto adding_setup = [](ModuleInfo& info) -> bool {
        TypeRef N = info.create_int(5);
        TypeRef M = info.create_int(6);
        if (!N || !M)
            return true;
        TypeRef fp = info.create_fty(core::EdgeFty::F16);
        TypeRef shape = info.create_array({ N, M });
        if (!fp || !shape)
            return true;
        return info.entry_setup("model", { {"fp", fp}, {"shape", shape} });
    };
    if (generate_graph("adding.nn", "adding.bcnn", graph, adding_setup))
        return true;

    // Attaching an SGD optimizer
    std::vector<GraphMod> opt;
    for (const auto& [name, param] : graph.weights)
    {
        assert(param.forward->info == param.backward->info);
        auto sgd_setup_fn = [&](ModuleInfo& info) -> bool {
            TypeRef lr = info.create_float(1e-3);
            TypeRef fp = info.create_fty(param.forward->info.fty);
            if (!lr || !fp)
                return true;
            std::vector<TypeRef> elems;
            for (size_t dim : param.forward->info.dims)
            {
                elems.push_back(info.create_int(dim));
                if (!elems.back())
                    return true;
            }
            TypeRef shape = info.create_array(elems);
            if (!shape)
                return true;
            return info.entry_setup("SGD", { { "lr", lr }, {"fp", fp}, {"shape", shape} });
        };
        core::Graph sgd;
        if (generate_graph("adding.nn", "", sgd, sgd_setup_fn))
            return true;
        opt.push_back({ {
            .inp_map = {{{core::InpRef::Type::WEIGHT, name}, {core::OutRef::Type::OUTPUT, "weight"}}},
            .out_map = {}
        }, sgd });
    }

    if (attach_graph(graph, "training", opt))
        return true;

    core::MdGraph md_graph{ "" };
    if (md_graph.init(graph))
        return true;

    return false;
}

bool test_simple()
{
    core::Graph graph;
    auto adding_setup = [](ModuleInfo& info) -> bool {
        TypeRef N = info.create_int(5);
        TypeRef fp = info.create_fty(core::EdgeFty::F16);
        if (!fp || !N) return true;
        return info.entry_setup("model", { { "fp", fp }, { "N", N } });
    };
    return generate_graph("simple.nn", "simple.bcnn", graph, adding_setup);
}

bool test_basic_structs()
{
    core::Graph graph;
    auto setup = [](ModuleInfo& info) -> bool {
        return info.entry_setup("model", {});
    };
    return generate_graph("structs/basic.nn", "basic_structs.bcnn", graph, setup);
}

bool test_recursive_structs()
{
    core::Graph graph;
    auto setup = [](ModuleInfo& info) -> bool {
        return info.entry_setup("model", {});
    };
    return generate_graph("structs/recursive.nn", "recursive_structs.bcnn", graph, setup);
}

bool test_generic_structs()
{
    core::Graph graph;
    auto setup = [](ModuleInfo& info) -> bool {
        return info.entry_setup("model", {});
    };
    return generate_graph("structs/generic.nn", "generic_structs.bcnn", graph, setup);
}

bool test_lang()
{
    core::Graph graph;
    auto setup = [](ModuleInfo& info) -> bool {
        return info.entry_setup("model", {});
    };
    return generate_graph("lang.nn", "lang.bcnn", graph, setup);
}

int main(void)
{
    if (test_lang())
    {
        error::print();
        return 1;
    }
    return 0;
}
