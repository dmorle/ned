#define _CRT_SECURE_NO_WARNINGS

#include <iostream>

#include <ned/errors.h>
#include <ned/lang/bytecode.h>
#include <ned/lang/interp.h>

using namespace nn;
using namespace lang;
using namespace core;

// too big to fit on the actual stack
CallStack stack;

int main()
{
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
    
    ByteCode byte_code;
    GraphBuilder builder;
    Obj obj, fwd, bwd;

    if (mod.export_module(byte_code) ||
        // creating the cargs
        heap.create_obj_fwidth(obj, EdgeFty::F32) ||
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

    return 0;
}
