#include <libnn/frontend/ast.h>
#include <libnn/frontend/obj.h>

namespace nn
{
    namespace frontend
    {
        Module::Module(const TokenArray& tarr)
        {

            for (int i = 0; i < tarr.size(); i++)
            {
                if (tarr[i]->ty == TokenType::ENDL || tarr[i]->ty == TokenType::INDENT)
                    continue;

                if (tarr[i]->ty != TokenType::IDN)
                    throw SyntaxError(tarr[i], "Invalid token");

                const char* idn_name = ((TokenImp<TokenType::IDN>*)tarr[i])->val;
                if (idn_name == "import")
                {
                    // TODO: figure out and implement importing
                }
                else if (idn_name == "def")
                {
                    int colon_pos = tarr.search<TokenArray::is_same<TokenType::COLON>>();
                    if (colon_pos < 0 || tarr.size() == colon_pos)
                        throw SyntaxError(tarr[i], "Invalid 'def'");
                    int block_end = tarr.search<TokenArray::block_end<0>>(colon_pos + 1);
                    if (block_end < 0)
                        throw SyntaxError(tarr[i], "Unable to find the block end for 'def'");

                    this->defs.push_back({
                        {tarr, i, colon_pos},             // signature
                        {tarr, colon_pos + 1, block_end}  // block
                    });

                    i = block_end - 1;
                }
                else
                    throw SyntaxError(tarr[i], "Invalid token");
            }
        }

        Obj* Module::eval(const std::string& entry_point, EvalCtx& ctx)
        {
        }

        Ast* parse_expr(const TokenArray& tarr, int lvl)
        {
        }
    }
}
