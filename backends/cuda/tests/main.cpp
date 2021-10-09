#include <array>
#include <iostream>
#include <type_traits>

#include <ned/lang/obj.h>
#include <cuned/cugraph.h>

using namespace nn;

template<size_t N>
std::ostream& operator<<(std::ostream& os, float(&data)[N])
{
    static_assert(N > 0);
    os << "[" << data[0];
    for (size_t i = 1; i < N; i++)
        os << ", " << data[i];
    os << "]";
    return os;
}

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
    /*try
    {*/
        pmod = new lang::AstModule{ tarr };
    /*}
    catch (lang::SyntaxError& err)
    {
        printf("%s\n", err.what());
        fclose(pf);
        exit(1);
    }*/

    lang::EvalCtx* pctx = pmod->eval("model", { lang::create_obj_int(1), lang::create_obj_int(4), lang::create_obj_int(1) });
    //lang::EvalCtx* pctx = pmod->eval("model", { lang::create_obj_int(2) });
    delete pmod;
    
    cuda::CuGraph* pgraph = new cuda::CuGraph(pctx->pgraph);
    delete pctx;

    cuda::RunId id = pgraph->generate_id();

    // forward pass
    float inp1[] = { 1, 2, 3, 4 };
    float inp2[] = { 4, 3, 2, 1 };
    pgraph->assign_input("inp1", inp1, sizeof(inp1), id);
    pgraph->assign_input("inp2", inp2, sizeof(inp2), id);
    pgraph->forward(id);
    float out[1];
    pgraph->get_output(0, out, sizeof(out));
    std::cout << out << std::endl;

    // backward pass
    float grad[] = { 1 };
    pgraph->assign_grad(0, grad, sizeof(grad), id);
    pgraph->backward(id);
    float inp1_grad[4];
    float inp2_grad[4];
    pgraph->get_grad("inp1", inp1_grad, sizeof(inp1_grad));
    pgraph->get_grad("inp2", inp2_grad, sizeof(inp2_grad));
    std::cout << inp1_grad << std::endl;
    std::cout << inp2_grad << std::endl;

    delete pgraph;
}
