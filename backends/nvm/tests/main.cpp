#include <iostream>
#include <fstream>
#include <functional>

#include <ned/errors.h>
#include <ned/lang/lexer.h>
#include <ned/lang/ast.h>
#include <ned/lang/compiler.h>
#include <ned/lang/obj.h>
#include <ned/lang/interp.h>
#include <ned/lang/bytecode.h>

#include <ned/core/config.h>
#include <ned/core/graph.h>
#include <ned/core/reduce.h>

#include <nvm/graphgen.h>
#include <nvm/runtime.h>

using namespace nn;
using namespace nn::lang;

CallStack stack;

template<typename T>
void print_vec(T* vec, size_t sz)
{
    for (size_t i = 0; i < sz - 1; i++)
        std::cout << vec[i] << ", ";
    std::cout << vec[sz - 1] << std::endl;
}

template<typename T>
void print_mrx(T* vec, size_t m, size_t n)
{
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n - 1; j++)
        {
            T e = vec[i * n + j];
            std::cout << e << ", ";
        }
        std::cout << vec[i * n + n - 1] << std::endl;
    }
}

bool generate_graph(core::Graph& graph, std::function<bool(ModuleInfo&)> setup)
{
    TokenArray tarr;
    if (lex_file(TESTS_DIR"test.nn", tarr))
        return true;

    AstModule ast;
    if (parse_module(tarr, ast))
        return true;

    ProgramHeap heap;
    ByteCodeModule bc{ heap };
    ModuleInfo info;
    if (codegen_module(bc, info, ast, { TESTS_DIR }))
        return true;

    TypeManager manager{};
    info.init(&manager);
    if (setup(info))
        return true;

    std::ofstream ofs{ TESTS_DIR"bcdump/test.bcnn" };
    ofs << bc.to_string() << std::endl;

    ByteCode byte_code;
    GraphBuilder builder;
    return
        bc.export_module(byte_code) ||
        exec(stack, heap, builder, byte_code, "main") ||
        builder.export_graph(graph);
}

bool compile_graph(core::Graph& graph, nvm::Runtime& runtime)
{
    core::MdGraph md_graph("");
    if (md_graph.init(graph))
        return true;

    nvm::GraphCompiler graph_comp;
    if (graph_comp.init(md_graph))
        return true;

    if (graph_comp.compile())
        return true;
    return runtime.init("test.dll");
}

bool vecadd_test()
{
    core::Graph graph;
    auto setup_fn = [](ModuleInfo& info) -> bool {
        TypeRef N = info.create_int(3);
        if (!N)
            return true;
        TypeRef fp = info.create_fty(core::EdgeFty::F32);
        TypeRef shape = info.create_array({ N });
        if (!fp || !shape)
            return true;
        return info.entry_setup("add_model", { {"fp", fp}, {"shape", shape} });
    };
    if (generate_graph(graph, setup_fn))
        return true;
    nvm::Runtime runtime;
    if (compile_graph(graph, runtime))
        return true;

    float lhs[3] = { 1, 2, 3 };
    float rhs[3] = { 2, 2, 2 };
    float out[3] = {};

    if (runtime.set_inp("lhs", (uint8_t*)lhs) ||
        runtime.set_inp("rhs", (uint8_t*)rhs)
        ) return true;

    runtime.run();
    if (runtime.get_out("out", (uint8_t*)out))
        return true;

    std::cout << "Vector Addition test:" << std::endl;
    std::cout << "lhs" << std::endl;
    print_vec(lhs, 3);
    std::cout << "rhs" << std::endl;
    print_vec(rhs, 3);
    std::cout << "out" << std::endl;
    print_vec(out, 3);

    return false;
}

bool transpose_test()
{
    core::Graph graph;
    auto setup_fn = [](ModuleInfo& info) -> bool {
        TypeRef fp = info.create_fty(core::EdgeFty::F32);
        TypeRef sz = info.create_int(2);
        if (!fp || !sz)
            return true;
        return info.entry_setup("tr_model", { {"fp", fp}, {"M", sz}, {"N", sz} });
    };
    if (generate_graph(graph, setup_fn))
        return true;
    nvm::Runtime runtime;
    if (compile_graph(graph, runtime))
        return true;

    float inp[4] = { 1, 2, 1, 2 };
    float out[4] = {};

    if (runtime.set_inp("inp", (uint8_t*)inp))
        return true;

    runtime.run();
    if (runtime.get_out("out", (uint8_t*)out))
        return true;

    std::cout << "Transpose test:" << std::endl;
    std::cout << "inp" << std::endl;
    print_mrx(inp, 2, 2);
    std::cout << "out" << std::endl;
    print_mrx(out, 2, 2);

    return false;
}

bool matmul_test()
{
    core::Graph graph;
    auto setup_fn = [](ModuleInfo& info) -> bool {
        TypeRef fp = info.create_fty(core::EdgeFty::F32);
        TypeRef sz = info.create_int(2);
        if (!fp || !sz)
            return true;
        return info.entry_setup("mm_model", { {"fp", fp}, {"M", sz}, {"K", sz}, {"N", sz} });
    };
    if (generate_graph(graph, setup_fn))
        return true;
    nvm::Runtime runtime;
    if (compile_graph(graph, runtime))
        return true;

    float lhs[4] = { 1, 2, 1, 2 };
    float rhs[4] = { 0, 1, 1, 0 };
    float out[4] = {};

    if (runtime.set_inp("lhs", (uint8_t*)lhs) ||
        runtime.set_inp("rhs", (uint8_t*)rhs)
        ) return true;

    runtime.run();
    if (runtime.get_out("out", (uint8_t*)out))
        return true;

    std::cout << "Matmul test:" << std::endl;
    std::cout << "lhs" << std::endl;
    print_mrx(lhs, 2, 2);
    std::cout << "rhs" << std::endl;
    print_mrx(rhs, 2, 2);
    std::cout << "out" << std::endl;
    print_mrx(out, 2, 2);

    return false;
}

bool linear_test()
{
    core::Graph graph;
    auto setup_fn = [](ModuleInfo& info) -> bool {
        TypeRef fp = info.create_fty(core::EdgeFty::F32);
        TypeRef sz = info.create_int(2);
        if (!fp || !sz)
            return true;
        return info.entry_setup("lin_model", { {"fp", fp}, {"M", sz}, {"K", sz}, {"N", sz} });
    };
    if (generate_graph(graph, setup_fn))
        return true;
    nvm::Runtime runtime;
    if (compile_graph(graph, runtime))
        return true;

    float lhs[4] = { 1, 2, 1, 2 };
    float rhs[4] = { 0, 1, 1, 0 };
    float bias[4] = { 1, 1, 2, 2 };
    float out[4] = {};

    if (runtime.set_inp("lhs", (uint8_t*)lhs) ||
        runtime.set_inp("rhs", (uint8_t*)rhs) ||
        runtime.set_inp("bias", (uint8_t*)bias)
        ) return true;

    runtime.run();
    if (runtime.get_out("out", (uint8_t*)out))
        return true;

    std::cout << "Linear test:" << std::endl;
    std::cout << "lhs" << std::endl;
    print_mrx(lhs, 2, 2);
    std::cout << "rhs" << std::endl;
    print_mrx(rhs, 2, 2);
    std::cout << "bias" << std::endl;
    print_mrx(bias, 2, 2);
    std::cout << "out" << std::endl;
    print_mrx(out, 2, 2);

    return false;
}

bool optimizer_test()
{
    auto setup_fn = [](ModuleInfo& info) -> bool {
        TypeRef fp = info.create_fty(core::EdgeFty::F32);
        TypeRef sz = info.create_int(2);
        if (!fp || !sz)
            return true;
        return info.entry_setup("opt_model", { {"fp", fp}, {"N", sz}, {"inp_sz", sz}, {"out_sz", sz} });
    };
    core::Graph graph;
    if (generate_graph(graph, setup_fn))
        return true;

    std::vector<core::GraphMod> opt;
    for (const auto& [name, param] : graph.weights)
    {
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
        if (generate_graph(sgd, sgd_setup_fn))
            return true;
        opt.push_back({ {
            .inp_map = {{{core::InpRef::Type::WEIGHT, name}, {core::OutRef::Type::OUTPUT, "weight"}}},
            .out_map = {}
        }, sgd });
    }

    if (attach_graph(graph, "training", opt))
        return true;
    nvm::Runtime runtime;
    if (compile_graph(graph, runtime))
        return true;
    return false;
}

int main()
{
    if (vecadd_test())
    {
        error::print();
        return 1;
    }
    return 0;
}
