#define _CRT_SECURE_NO_WARNINGS

#include <iostream>

#include <ned/lang/lexer.h>
#include <ned/lang/ast.h>
#include <ned/lang/obj.h>

using namespace nn;
using namespace lang;

int main()
{
    FILE* pf = fopen(TESTS_DIR"test.nn", "rb");
    TokenArray tarr{};
    lex_file(pf, tarr);
    AstModule mod{ tarr };
    EvalCtx* pctx = mod.eval("model", { create_obj_int(10) });
}
