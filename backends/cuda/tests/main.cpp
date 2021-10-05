#include <iostream>

#include <ned/lang/obj.h>
#include <cuned/cugraph.h>

using namespace nn;

int main()
{
    FILE* pf = fopen(TEST_DIR"test.nn", "rb");
    lang::TokenArray tarr{};
    try
    {
        lang::lex_file(pf, tarr);
    }
    catch (lang::SyntaxError & err)
    {
        printf("%s\n", err.what());
        fclose(pf);
        exit(1);
    }

    fclose(pf);

    lang::AstModule* pmod;
    try
    {
        pmod = new lang::AstModule{ tarr };
    }
    catch (lang::SyntaxError& err)
    {
        printf("%s\n", err.what());
        fclose(pf);
        exit(1);
    }

    lang::EvalCtx* pctx = pmod->eval("model", { lang::create_obj_int(2) });
    delete pmod;
    
    cuda::CuGraph* pgraph = new cuda::CuGraph(pctx->pgraph);
    delete pctx;

    cuda::RunId id = pgraph->generate_id();

    float inp1[] = { 1, 2 };
    float inp2[] = { 2, 3 };
    pgraph->assign_input("inp1", inp1, sizeof(inp1), id);
    pgraph->assign_input("inp2", inp2, sizeof(inp2), id);
    pgraph->forward(id);
    float out[2];
    pgraph->get_output(0, out, sizeof(out));

    for (float val : out)
        std::cout << val << ", ";
    std::cout << std::endl;

    delete pgraph;
}
