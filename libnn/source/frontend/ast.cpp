#include <libnn/frontend/ast.h>
#include <libnn/frontend/obj.h>

namespace nn
{
    namespace frontend
    {
        bool isDecl(const TokenArray& tarr)
        {

        }

        AstExpr* parseExpr(const TokenArray& tarr)
        {
        }

        // helper for function signatures ie. def my_func<...>(...)
        void parseDeclList(const TokenArray& tarr, std::vector<AstDecl>& declList)
        {
            int start = 0;
            int end;
            do
            {
                end = tarr.search<TokenArray::is_same<TokenType::COMMA>>();
                TokenArray decl(tarr, start, end);
                declList.push_back({ decl });
                start = end;
            } while (end != -1);
        }

        AstSeq::AstSeq(const TokenArray& tarr, int indent_level)
        {
            // Assuming the first 'indent_level' tokens are TokenType::INDENT for every line
            int start = indent_level;
            int end;
            while (start < tarr.size())
            {
                if (tarr[start]->ty == TokenType::IDN)
                {
                    // checking for keywords
                    if (static_cast<const TokenImp<TokenType::IDN>*>(tarr[start])->val == "if")
                    {
                        int colon_pos = tarr.search<TokenArray::is_same<TokenType::COLON>>(start);
                        if (colon_pos < 0 || tarr.size() - 1 == colon_pos)
                            throw SyntaxError(tarr[start], "Invalid if statement");
                        end = tarr.search<TokenArray::block_end<0>>(colon_pos + 1);
                        if (end < 0)
                            end = tarr.size();  // the rest of the tokens make up the seq
                        if (colon_pos + 1 == end)
                            throw SyntaxError(tarr[colon_pos], "Empty code block following ':'");

                        TokenArray if_sig(tarr, start, colon_pos);
                        TokenArray if_seq(tarr, colon_pos + 1, end);
                        this->blocks.push_back(new AstIf(if_sig, if_seq, indent_level + 1));
                        // else isn't a thing yet...
                        start = end + indent_level;
                        continue;
                    }
                    else if (static_cast<const TokenImp<TokenType::IDN>*>(tarr[start])->val == "while")
                    {
                        int colon_pos = tarr.search<TokenArray::is_same<TokenType::COLON>>(start);
                        if (colon_pos < 0 || tarr.size() - 1 == colon_pos)
                            throw SyntaxError(tarr[start], "Invalid while loop");
                        end = tarr.search<TokenArray::block_end<0>>(colon_pos + 1);
                        if (end < 0)
                            end = tarr.size();  // the rest of the tokens make up the seq
                        if (colon_pos + 1 == end)
                            throw SyntaxError(tarr[colon_pos], "Empty code block following ':'");

                        TokenArray while_sig(tarr, start, colon_pos);
                        TokenArray while_seq(tarr, colon_pos + 1, end);
                        this->blocks.push_back(new AstWhile(while_sig, while_seq, indent_level + 1));
                        start = end + indent_level;
                        continue;
                    }
                    else if (static_cast<const TokenImp<TokenType::IDN>*>(tarr[start])->val == "for")
                    {
                        int colon_pos = tarr.search<TokenArray::is_same<TokenType::COLON>>(start);
                        if (colon_pos < 0 || tarr.size() - 1 == colon_pos)
                            throw SyntaxError(tarr[start], "Invalid for loop");
                        end = tarr.search<TokenArray::block_end<0>>(colon_pos + 1);
                        if (end < 0)
                            end = tarr.size();  // the rest of the tokens make up the seq
                        if (colon_pos + 1 == end)
                            throw SyntaxError(tarr[colon_pos], "Empty code block following ':'");

                        TokenArray for_sig(tarr, start, colon_pos);
                        TokenArray for_seq(tarr, colon_pos + 1, end);
                        this->blocks.push_back(new AstFor(for_sig, for_seq, indent_level + 1));
                        start = end + indent_level;
                        continue;
                    }
                }

                int assign_pos = tarr.search<TokenArray::is_same<TokenType::ASSIGN>>(start);
                if (assign_pos >= start + 2)  // minimum requirement for a decl, ie. int x = 1
                {
                    TokenArray decl_tarr(tarr, start, assign_pos);
                    if (isDecl(decl_tarr))
                    {
                        this->blocks.push_back(new AstDecl(decl_tarr));
                        start = assign_pos - 1;  // setup for parsing the expression seperately
                    }
                }

                end = tarr.search<TokenArray::is_same<TokenType::ENDL>>(start);
                if (end < 0)
                    end = tarr.size();
                TokenArray expr_tarr(tarr, start, end);
                this->blocks.push_back(parseExpr(expr_tarr));

                start = end + 1 + indent_level;  // eat the ENDL token and the 'indent_level' INDENT tokens.
            }
        }

        AstSeq::~AstSeq()
        {
            for (auto e : blocks)
                delete e;
        }

        AstDef::AstDef(const TokenArray& def_sig, const TokenArray& def_seq) :
            block(def_seq, 1)
        {
            // parsing the signature
            // def _() <- minimum allowable signature
            if (def_sig.size() < 4 || def_sig[0]->ty != TokenType::IDN || static_cast<const TokenImp<TokenType::IDN>*>(def_sig[0])->val != "def")
                throw SyntaxError(def_sig[0], "Invalid function signature");

            if (def_sig[1]->ty != TokenType::IDN)
                throw SyntaxError(def_sig[1], "Invalid function signature");
            this->name = static_cast<const TokenImp<TokenType::IDN>*>(def_sig[1])->val;

            int start = 2;
            int end;
            if (def_sig[start]->ty == TokenType::ANGLE_O)
            {
                end = def_sig.search<TokenArray::is_same<TokenType::ANGLE_C>>();
                if (end < 0)
                    throw SyntaxError(def_sig[start], "Missing closing '>' in function signature");
                TokenArray constargs_tarr({ def_sig, start, end });
                parseDeclList(constargs_tarr, this->constargs);
                start = end + 1;
                if (start >= def_sig.size())
                    throw SyntaxError(def_sig[end], "Missing expected '(' in function signature");
            }

            if (def_sig[start]->ty != TokenType::ROUND_O)
                throw SyntaxError(def_sig[start], "Missing expected '(' in function signature");

            end = def_sig.search<TokenArray::is_same<TokenType::ROUND_C>>();
            if (end < 0)
                throw SyntaxError(def_sig[start], "Missing closing ')' in function signature");

            TokenArray varargs_tarr({ def_sig, start, end });
            parseDeclList(varargs_tarr, this->varargs);
        }

        Obj* AstDef::eval(EvalCtx& ctx, Module& mod)
        {
            throw std::logic_error("Not implemented");
        }

        Module::Module(const TokenArray& tarr)
        {
            for (int i = 0; i < tarr.size(); i++)
            {
                if (tarr[i]->ty == TokenType::ENDL || tarr[i]->ty == TokenType::INDENT)
                    continue;

                if (tarr[i]->ty != TokenType::IDN)
                    throw SyntaxError(tarr[i], "Invalid token");

                const char* idn_name = static_cast<const TokenImp<TokenType::IDN>*>(tarr[i])->val;
                if (idn_name == "import")
                {
                    // TODO: figure out and implement importing
                }
                else if (idn_name == "def")
                {
                    int colon_pos = tarr.search<TokenArray::is_same<TokenType::COLON>>();
                    if (colon_pos < 1 || tarr.size() == colon_pos)
                        throw SyntaxError(tarr[i], "Invalid 'def'");
                    int block_end = tarr.search<TokenArray::block_end<0>>(colon_pos + 1);
                    if (block_end < 0)
                        block_end = tarr.size();  // the rest of the tokens make up the block
                    if (colon_pos + 1 == block_end)
                        throw SyntaxError(tarr[colon_pos], "Empty code block following ':'");

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
            throw std::logic_error("Not implemented");
        }
    }
}
