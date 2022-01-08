#define _CRT_SECURE_NO_WARNINGS

#include <iostream>

#include <ned/lang/errors.h>
#include <ned/lang/bytecode.h>
#include <ned/lang/interp.h>

using namespace nn::lang;

// too big to fit on the c++ stack
CallStack stack;

int main()
{
    FILE* pf = fopen(TESTS_DIR"test.bcnn", "rb");
    Errors errs;
    ProgramHeap heap;
    ByteCodeModule mod{ heap };
    if (parsebc_module(errs, TESTS_DIR"test.bcnn", pf, mod))
    {
        errs.print();
        return 1;
    }
    fclose(pf);
    CodeSegPtr code;
    DataSegPtr data;
    BlockOffsets offsets;
    Obj obj;

    if (mod.export_module(errs, code, data, offsets) ||
        heap.create_obj_int(errs, obj, 5) ||
        stack.push(errs, obj) ||
        heap.create_obj_int(errs, obj, 3) ||
        stack.push(errs, obj) ||
        exec(errs, stack, heap, code, data, offsets.at("test")))
    {
        errs.print();
        return 1;
    }

    //lex_file(TESTS_DIR"test.nn", pf, tarr);
    //AstModule* pmod = new AstModule{ tarr };
    //EvalCtx* pctx = pmod->eval("model", { create_obj_int(10) });
    //delete pctx;
    //delete pmod;
}
