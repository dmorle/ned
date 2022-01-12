#define _CRT_SECURE_NO_WARNINGS

#include <iostream>

#include <ned/lang/errors.h>
#include <ned/lang/bytecode.h>
#include <ned/lang/interp.h>

using namespace nn::lang;
using namespace nn::core;

// too big to fit on the actual stack
CallStack stack;

int main()
{
    Errors errs;
    TokenArray tarr;
    if (lex_file(errs, TESTS_DIR"test.bcnn", tarr))
    {
        errs.print();
        return 1;
    }

    ProgramHeap heap;
    ByteCodeModule mod{ heap };
    if (parsebc_module(errs, tarr, mod))
    {
        errs.print();
        return 1;
    }
    
    ByteCode byte_code;
    Graph graph;
    Obj obj;

    if (mod.export_module(errs, byte_code) ||
        heap.create_obj_int(errs, obj, 100000) ||
        stack.push(errs, obj) ||
        exec(errs, stack, heap, byte_code, "test", graph))
    {
        errs.print();
        return 1;
    }
}
