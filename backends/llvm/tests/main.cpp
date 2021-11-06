#include <iostream>

#include <ned/lang/obj.h>
#include <nedvm/graphgen.h>

using namespace nn;


int main()
{
    FILE* pf = fopen(TEST_DIR"test.nn", "rb");
    lang::TokenArray tarr{};
    try
    {
        lang::lex_file(pf, tarr);
    }
    catch (lang::SyntaxError& err)
    {
        printf("%s\n", err.what());
        fclose(pf);
        exit(1);
    }

    fclose(pf);

    lang::AstModule* pmod = new lang::AstModule{ tarr };
    lang::EvalCtx* pctx = pmod->eval("model", { lang::create_obj_int(5) });
    delete pmod;
    nedvm::GraphCompiler* pcompiler = new nedvm::GraphCompiler(pctx->pgraph);
    pcompiler->generate_forward();
    pcompiler->print();
    delete pctx;
    delete pcompiler;
}
