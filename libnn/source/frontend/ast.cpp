#include <libnn/frontend/ast.h>
#include <libnn/frontend/obj.h>

#include <cassert>

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
        void paseDefArgs(const TokenArray& tarr, std::vector<AstDefArgSingle>& args)
        {
            int start = 0;
            int end;
            do
            {
                end = tarr.search<TokenArray::args_elem>(start);
                TokenArray decl(tarr, start, end);
                args.push_back({ decl });
                start = end;
            } while (end != -1);
        }

        // helper for function signatures ie. def my_func<...>(...)
        void parseDefCargs(const TokenArray& tarr, std::vector<AstDefCarg*>& cargs)
        {
            if (tarr.size() == 0)
                return;

            int start = 0;
            int end;
            do
            {
                end = tarr.search<TokenArray::args_elem>(start);
                TokenArray carg(tarr, start, end);
                if (carg.size() == 0)
                    throw SyntaxError(tarr[0], "Dangling end of constargs in def");
                if (carg[0]->ty == TokenType::ANGLE_O)
                {
                    if (carg[carg.size() - 1]->ty != TokenType::ANGLE_C)
                        throw SyntaxError(carg[0], "Missing closing '>' for constarg tuple in def");
                    TokenArray cargtuple(carg, 1, -1);
                    cargs.push_back(new AstDefCargTuple(carg));
                }
                else
                    cargs.push_back(new AstDefArgSingle(carg));
                start = end + 1;
            } while (end != -1);
        }

        // helper for function calls ie. my_func<...>(...)
        void parseConstargs(const TokenArray& tarr, std::vector<AstExpr*>& exprs)
        {
            if (tarr.size() == 0)
                return;

            int start = 0;
            int end;
            do
            {
                end = tarr.search<TokenArray::args_elem>(start);
                TokenArray carg(tarr, start, end);
                if (carg.size() == 0)
                    throw SyntaxError(tarr[0], "Dangling end of constargs in block");
                if (carg[0]->ty == TokenType::ANGLE_O)
                {
                    if (carg[carg.size() - 1]->ty != TokenType::ANGLE_C)
                        throw SyntaxError(carg[0], "Missing closing '>' for constarg tuple in block");
                    TokenArray cargtuple(carg, 1, -1);
                    exprs.push_back(new AstTuple(cargtuple));
                }
                else
                    exprs.push_back(parseExpr(carg));
                start = end + 1;
            } while (end != -1);
        }

        AstDecl::AstDecl()
        {
            this->var_name = "";
            this->type_name = "";
            this->constargs = {};
        }

        AstDecl::AstDecl(const TokenArray& tarr)
        {
            assert(tarr.size());
            if (tarr.size() < 2)
                throw SyntaxError(tarr[0], "Invalid syntax for declaration");

            if (tarr[0]->ty != TokenType::IDN)
                throw SyntaxError(tarr[0], "Invalid token for type name in declaration");
            this->type_name = static_cast<const TokenImp<TokenType::IDN>*>(tarr[0])->val;
            if (this->type_name == "var")
            {
                if (tarr.size() != 2)
                    throw SyntaxError(tarr[0], "Invalid syntax for var declaration");
                if (tarr[1]->ty != TokenType::IDN)
                    throw SyntaxError(tarr[1], "Invalid token for variable name in var declaration");
                this->var_name = static_cast<const TokenImp<TokenType::IDN>*>(tarr[1])->val;
            }
            else
            {
                int var_idx = tarr.size() - 1;
                if (tarr[var_idx]->ty != TokenType::IDN)
                    throw SyntaxError(tarr[var_idx], "Invalid token for variable name in declaration");
                this->var_name = static_cast<const TokenImp<TokenType::IDN>*>(tarr[var_idx])->val;
                
                if (tarr.size() != 2)
                {
                    if (tarr[2]->ty != TokenType::ANGLE_O)
                        throw SyntaxError(tarr[2], "Expected '<' in type for variable declaration");
                    if (tarr[tarr.size() - 2]->ty != TokenType::ANGLE_C)
                        throw SyntaxError(tarr[tarr.size() - 2], "Expected '>' in type for variable declaration");
                    TokenArray cargs(tarr, 3, tarr.size() - 2);
                    parseConstargs(cargs, this->constargs);
                }
            }
        }

        AstIf::AstIf(const TokenArray& if_sig, const TokenArray& if_seq, int indent_level) :
            seq(if_seq, indent_level + 1)
        {
            assert(if_sig[0]->ty == TokenType::IDN);
            assert(static_cast<const TokenImp<TokenType::IDN>*>(if_sig[0])->val == "if");

            if (if_sig.size() < 2)  // if cond  <- minimum number of tokens in an if statement signature
                throw SyntaxError(if_sig[0], "Invalid if statement signature");

            this->pcond = parseExpr(TokenArray(if_sig, 1));
        }

        AstIf::~AstIf()
        {
            if (pcond)
                delete pcond;
        }

        Obj* AstIf::eval(EvalCtx& ctx, Module& mod)
        {
            throw std::logic_error("Not implemented");
        }

        AstWhile::AstWhile(const TokenArray& while_sig, const TokenArray& while_seq, int indent_level) :
            seq(while_seq, indent_level + 1)
        {
            assert(while_sig[0]->ty == TokenType::IDN);
            assert(static_cast<const TokenImp<TokenType::IDN>*>(while_sig[0])->val == "while");

            if (while_sig.size() < 2)  // while cond  <- minimum number of tokens in a while loop signature
                throw SyntaxError(while_sig[0], "Invalid while loop signature");

            this->pcond = parseExpr(TokenArray(while_sig, 1));
        }

        AstWhile::~AstWhile()
        {
            if (pcond)
                delete pcond;
        }

        Obj* AstWhile::eval(EvalCtx& ctx, Module& mod)
        {
            throw std::logic_error("Not implemented");
        }

        AstFor::AstFor(const TokenArray& for_sig, const TokenArray& for_seq, int indent_level) :
            seq(for_seq, indent_level + 1)
        {
            assert(for_sig[0]->ty == TokenType::IDN);
            assert(static_cast<const TokenImp<TokenType::IDN>*>(for_sig[0])->val == "for");

            if (for_sig.size() < 5)  // for var i in lst  <- minimum number of tokens in a for loop signature
                throw SyntaxError(for_sig[0], "Invalid for loop signature");

            int in_pos = for_sig.search<TokenArray::is_keyword<Keyword::IN>>(2);
            if (in_pos < 0)
                throw SyntaxError(for_sig[1], "Missing keyword in for loop signature: 'in'");
            this->it = TokenArray(for_sig, 1, in_pos);  // decl init
            this->pexpr = parseExpr(TokenArray(for_sig, in_pos + 1));
        }

        AstFor::~AstFor()
        {
            if (pexpr)
                delete pexpr;
        }

        Obj* AstFor::eval(EvalCtx& ctx, Module& mod)
        {
            throw std::logic_error("Not implemented");
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
                        this->blocks.push_back(new AstIf(if_sig, if_seq, indent_level));
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
                        this->blocks.push_back(new AstWhile(while_sig, while_seq, indent_level));
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
                        this->blocks.push_back(new AstFor(for_sig, for_seq, indent_level));
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

        Obj* AstSeq::eval(EvalCtx& ctx, Module& mod)
        {
            throw std::logic_error("Not implemented");
        }

        AstDefArgSingle::AstDefArgSingle(const TokenArray& tarr)
        {
            assert(tarr.size());
            if (tarr.size() < 2)
                throw SyntaxError(tarr[0], "Invalid syntax for declaration");

            if (tarr[0]->ty != TokenType::IDN)
                throw SyntaxError(tarr[0], "Invalid token for type name in declaration");
            this->type_name = static_cast<const TokenImp<TokenType::IDN>*>(tarr[0])->val;
            if (this->type_name == "var")
            {
                if (tarr[1]->ty == TokenType::STAR)
                {
                    packed = true;
                    if (tarr.size() != 3)
                        throw SyntaxError(tarr[0], "Invalid syntax for var* declaration");
                    if (tarr[2]->ty != TokenType::IDN)
                        throw SyntaxError(tarr[1], "Invalid token for variable name in var* declaration");
                    this->var_name = static_cast<const TokenImp<TokenType::IDN>*>(tarr[2])->val;
                }
                else
                {
                    packed = false;
                    if (tarr.size() != 2)
                        throw SyntaxError(tarr[0], "Invalid syntax for var declaration");
                    if (tarr[1]->ty != TokenType::IDN)
                        throw SyntaxError(tarr[1], "Invalid token for variable name in var declaration");
                    this->var_name = static_cast<const TokenImp<TokenType::IDN>*>(tarr[1])->val;
                }
            }
            else
            {
                if (tarr[tarr.size() - 1]->ty != TokenType::IDN)
                    throw SyntaxError(tarr[tarr.size() - 1], "Invalid token for variable name in declaration");
                this->var_name = static_cast<const TokenImp<TokenType::IDN>*>(tarr[tarr.size() - 1])->val;

                if (tarr[tarr.size() - 2]->ty == TokenType::STAR)
                {
                    packed = true;
                    if (tarr.size() != 3)
                    {
                        if (tarr[2]->ty != TokenType::ANGLE_O)
                            throw SyntaxError(tarr[2], "Expected '<' in type for variable declaration");
                        if (tarr[tarr.size() - 3]->ty != TokenType::ANGLE_C)
                            throw SyntaxError(tarr[tarr.size() - 3], "Expected '>' in type for variable declaration");
                        TokenArray cargs(tarr, 3, tarr.size() - 3);  // T<>*N
                        parseConstargs(cargs, this->constargs);
                    }
                }
                else
                {
                    packed = false;
                    if (tarr.size() != 2)
                    {
                        if (tarr[2]->ty != TokenType::ANGLE_O)
                            throw SyntaxError(tarr[2], "Expected '<' in type for variable declaration");
                        if (tarr[tarr.size() - 2]->ty != TokenType::ANGLE_C)
                            throw SyntaxError(tarr[tarr.size() - 2], "Expected '>' in type for variable declaration");
                        TokenArray cargs(tarr, 3, tarr.size() - 2);  // T<>N
                        parseConstargs(cargs, this->constargs);
                    }
                }
            }
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
                // TODO: allow for nested <>
                end = def_sig.search<TokenArray::constargs_end>();
                if (end < 0)
                    throw SyntaxError(def_sig[start], "Missing closing '>' in function signature");
                TokenArray constargs_tarr({ def_sig, start, end });
                parseDefCargs(constargs_tarr, this->constargs);
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
            paseDefArgs(varargs_tarr, this->varargs);
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
