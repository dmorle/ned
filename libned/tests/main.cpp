#define _CRT_SECURE_NO_WARNINGS

#include <iostream>

#include <ned/lang/lexer.h>
#include <ned/lang/ast.h>
#include <ned/lang/obj.h>

using namespace nn;
using namespace lang;

int main()
{
    for (int i = 0; i < 10000000; i++)
    {
        FILE* pf = fopen(TESTS_DIR"test.nn", "rb");
        TokenArray tarr{};
        lex_file(TESTS_DIR"test.nn", pf, tarr);
        fclose(pf);
        AstModule* pmod = new AstModule{ tarr };
        EvalCtx* pctx = pmod->eval("model", { create_obj_int(10) });
        delete pctx;
        delete pmod;
    }
}
