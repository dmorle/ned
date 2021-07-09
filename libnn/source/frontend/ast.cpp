#include <libnn/frontend/ast.h>
#include <libnn/frontend/obj.h>

namespace nn
{
    namespace frontend
    {
        Module::Module(const TokenArray& tarr)
        {
        }

        Obj* Module::eval(const std::string& entry_point, EvalCtx& ctx)
        {
        }

        Ast* parse_expr(const TokenArray& tarr, int lvl)
        {
        }
    }
}
