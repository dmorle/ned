#include <iostream>

#include <ned/lang/obj.h>
#include <cuned/cugraph.h>

using namespace nn;

int main()
{
    FILE* pf = fopen(TEST_DIR"test.nn", "rb");
    lang::TokenArray tarr{};
    lang::lex_file(pf, tarr);
    fclose(pf);
    lang::AstModule* pmod = new lang::AstModule{ tarr };
    lang::EvalCtx* pctx = pmod->eval("model", { lang::create_obj_int(1) });
    delete pmod;
    
    cuda::CuGraph* pgraph = new cuda::CuGraph(pctx->pgraph);
    delete pctx;

    cuda::RunId id = pgraph->generate_id();

    float inp1[] = { 1 };
    float inp2[] = { 2 };
    pgraph->assign_input("inp1", inp1, sizeof(inp1), id);
    pgraph->assign_input("inp2", inp2, sizeof(inp2), id);
    pgraph->eval(id);
    float out[1];
    pgraph->get_output(0, out, sizeof(out));

    delete pgraph;
}
