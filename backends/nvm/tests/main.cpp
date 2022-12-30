#include <iostream>
#include <fstream>

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

int main()
{
    TokenArray tarr;
    if (lex_file(TESTS_DIR"test.nn", tarr))
    {
        error::print();
        return 1;
    }

    AstModule ast;
    if (parse_module(tarr, ast))
    {
        error::print();
        return 1;
    }

    ProgramHeap heap;
    ByteCodeModule bc{ heap };
    ModuleInfo info;
    if (codegen_module(bc, info, ast, {}))
    {
        error::print();
        return 1;
    }

    TypeManager manager{};
    info.init(&manager);
    TypeRef N = info.create_int(2);
    TypeRef M = info.create_int(2);
    if (!N || !M)
    {
        error::print();
        return 1;
    }
    TypeRef fp = info.create_fty(core::EdgeFty::F32);
    TypeRef shape = info.create_array({ N, M });
    if (!fp || !shape || info.entry_setup("model", { {"fp", fp}, {"shape", shape} }))
    {
        error::print();
        return 1;
    }

    std::ofstream ofs{ TESTS_DIR"bcdump/test.bcnn" };
    ofs << bc.to_string() << std::endl;

    ByteCode byte_code;
    GraphBuilder builder;
    if (bc.export_module(byte_code) || exec(stack, heap, builder, byte_code, "main"))
    {
        error::print();
        return 1;
    }

    core::Graph graph;
    if (builder.export_graph(graph))
    {
        error::print();
        return 1;
    }

    core::MdGraph md_graph("");
    if (md_graph.init(graph))
    {
        error::print();
        return 1;
    }

    nvm::GraphCompiler graph_comp;
    if (graph_comp.init(md_graph))
    {
        error::print();
        return 1;
    }

    if (graph_comp.compile())
    {
        error::print();
        return 1;
    }

    float lhs[4] = { 1, 2, 3, 4 };
    float rhs[4] = { 2, 2, 2, 2 };
    float out[4] = {};

    nvm::Runtime runtime;
    if (runtime.init("test.dll"))
    {
        error::print();
        return 1;
    }

    if (runtime.set_inp("lhs", (uint8_t*)lhs) ||
        runtime.set_inp("rhs", (uint8_t*)rhs))
    {
        error::print();
        return 1;
    }

    runtime.step();
    if (runtime.get_out("out", (uint8_t*)out))
    {
        error::print();
        return 1;
    }

    print_vec(lhs, 4);
    print_vec(rhs, 4);
    print_vec(out, 4);

    return 0;
}
