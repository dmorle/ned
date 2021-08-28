#include <iostream>

#include <ned/lang/lexer.h>
#include <ned/lang/ast.h>
#include <ned/lang/obj.h>

using namespace nn;
using namespace impl;

int main()
{
    FILE* pf = fopen(TESTS_DIR"test.nn", "rb");
    TokenArray tarr{};
    lex_file(pf, tarr);
    AstModule mod{ tarr };
    EvalCtx* pctx = mod.eval("model", { impl::create_obj_int(10) });
}
