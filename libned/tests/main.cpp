#include <iostream>

#include <ned/lang/lexer.h>
#include <ned/lang/ast.h>

using namespace nn;
using namespace impl;

int main()
{
    FILE* pf = fopen(TESTS_DIR"test.nn", "rb");
    TokenArray tarr{};
    lex_file(pf, tarr);
    AstModule mod{ tarr };
    EvalCtx* pctx = mod.eval("model", {});

    std::cout << "Hello" << std::endl;
}
