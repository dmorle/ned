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
    if (lex_file(TESTS_DIR"test.bcnn", tarr))
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
    Graph graph;
    Obj obj;

    if (mod.export_module(byte_code) ||
        heap.create_obj_int(obj, 1000) ||
        stack.push(obj) ||
        exec(stack, heap, byte_code, "test", graph))
    {
        error::print();
        return 1;
    }
}
