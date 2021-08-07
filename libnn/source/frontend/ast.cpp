#include <libnn/frontend/ast.h>
#include <libnn/frontend/obj.h>

#include <cassert>

namespace nn
{
    namespace impl
    {
        bool isDecl(const TokenArray& tarr)
        {
            if (tarr.size() < 2 || tarr[0]->ty != TokenType::IDN)
                return false;

            size_t pos = 1;
            if (static_cast<const TokenImp<TokenType::IDN>*>(tarr[0])->val == "static")
            {
                if (tarr.size() < 3 || tarr[1]->ty != TokenType::IDN)
                    return false;
                pos = 2;
            }

            if (tarr[pos]->ty == TokenType::IDN)
                return true;

            if (tarr.size() < pos + 3 || tarr[pos]->ty != TokenType::ANGLE_O)  // [static] ty<> name
                return false;

            int ret = tarr.search<TokenArray::brac_end<TokenType::ANGLE_O, TokenType::ANGLE_C>>(pos + 1);
            if (ret < 0)
                throw SyntaxError(tarr[pos], "Missing closing '>'");
            assert(tarr[ret]->ty == TokenType::ANGLE_C);

            return !(ret + 1 == tarr.size() || tarr[ret + 1]->ty != TokenType::IDN);
        }

        int opPrec(const Token* ptk)
        {
            switch (ptk->ty)
            {
            case TokenType::ADD:
            case TokenType::SUB:
                return 3;
            case TokenType::STAR:
            case TokenType::DIV:
                return 4;
            case TokenType::CMP_EQ:
            case TokenType::CMP_NE:
            case TokenType::CMP_GE:
            case TokenType::CMP_LE:
            case TokenType::ANGLE_C:  // less than
            case TokenType::ANGLE_O:  // greater than
                return 2;
            case TokenType::IADD:
            case TokenType::ISUB:
            case TokenType::IMUL:
            case TokenType::IDIV:
            case TokenType::ASSIGN:
                return 0;
            case TokenType::IDN:
                if (static_cast<const TokenImp<TokenType::IDN>*>(ptk)->val == "and")
                    return 1;
                if (static_cast<const TokenImp<TokenType::IDN>*>(ptk)->val == "or")
                    return 1;
                return -1;
            }
            return -1;
        }

        template<int prec>
        AstExpr* splitExpr(const TokenArray& tarr, int split_point);

        template<>
        AstExpr* splitExpr<0>(const TokenArray& tarr, int split_point)
        {
            assert(split_point < tarr.size() - 1);

            TokenArray left(tarr, 0, split_point);
            TokenArray right(tarr, split_point + 1);

            switch (tarr[split_point]->ty)
            {
            case TokenType::IADD:
                return new AstIAdd(left, right);
            case TokenType::ISUB:
                return new AstISub(left, right);
            case TokenType::IMUL:
                return new AstIMul(left, right);
            case TokenType::IDIV:
                return new AstIDiv(left, right);
            case TokenType::ASSIGN:
                return new AstAssign(left, right);
            }

            assert(false);
        }

        template<>
        AstExpr* splitExpr<1>(const TokenArray& tarr, int split_point)
        {
            assert(split_point < tarr.size() - 1);
            assert(tarr[split_point]->ty == TokenType::IDN);

            TokenArray left(tarr, 0, split_point);
            TokenArray right(tarr, split_point + 1);

            if (static_cast<const TokenImp<TokenType::IDN>*>(tarr[split_point])->val == "and")
                return new AstAnd(left, right);
            if (static_cast<const TokenImp<TokenType::IDN>*>(tarr[split_point])->val == "or")
                return new AstOr(left, right);

            assert(false);
        }

        template<>
        AstExpr* splitExpr<2>(const TokenArray& tarr, int split_point)
        {
            assert(split_point < tarr.size() - 1);
            
            TokenArray left(tarr, 0, split_point);
            TokenArray right(tarr, split_point + 1);

            switch (tarr[split_point]->ty)
            {
            case TokenType::CMP_EQ:
                return new AstEq(left, right);
            case TokenType::CMP_NE:
                return new AstNe(left, right);
            case TokenType::CMP_GE:
                return new AstGe(left, right);
            case TokenType::CMP_LE:
                return new AstLe(left, right);
            case TokenType::ANGLE_C:
                return new AstGt(left, right);
            case TokenType::ANGLE_O:
                return new AstLt(left, right);
            }

            assert(false);
        }

        template<>
        AstExpr* splitExpr<3>(const TokenArray& tarr, int split_point)
        {
            assert(split_point < tarr.size() - 1);

            TokenArray left(tarr, 0, split_point);
            TokenArray right(tarr, split_point + 1);

            switch (tarr[split_point]->ty)
            {
            case TokenType::ADD:
                return new AstAdd(left, right);
            case TokenType::SUB:
                return new AstSub(left, right);
            }

            assert(false);
        }

        template<>
        AstExpr* splitExpr<4>(const TokenArray& tarr, int split_point)
        {
            assert(split_point < tarr.size() - 1);

            TokenArray left(tarr, 0, split_point);
            TokenArray right(tarr, split_point + 1);

            switch (tarr[split_point]->ty)
            {
            case TokenType::STAR:
                return new AstMul(left, right);
            case TokenType::DIV:
                return new AstDiv(left, right);
            }

            assert(false);
        }

        AstExpr* parseLeafMods(AstExpr* pleft, const TokenArray& tarr)
        {
            // parse member accesses, slices, function calls, etc.
            // ex: .val<4, 5>.test[0].fn<>()[::].h[5:]()
            if (tarr.size() == 0)
                return pleft;
            
            int end;
            switch (tarr[0]->ty)
            {
            case TokenType::ROUND_O:
                // function call
                end = tarr.search<TokenArray::brac_end<TokenType::ROUND_O, TokenType::ROUND_C>>(1);
                if (end < 0)
                    throw SyntaxError(tarr[0], "Missing closing ')'");
                return parseLeafMods(new AstCall(pleft, { tarr, 1, end }), { tarr, end + 1 });
            case TokenType::ANGLE_O:
                // constargs
                end = tarr.search<TokenArray::brac_end<TokenType::ANGLE_O, TokenType::ANGLE_C>>(1);
                if (end < 0)
                    throw SyntaxError(tarr[0], "Missing closing '>'");
                return parseLeafMods(new AstCargs(pleft, { tarr, 1, end }), { tarr, end + 1 });
            case TokenType::SQUARE_O:
                // slice
                end = tarr.search<TokenArray::brac_end<TokenType::SQUARE_O, TokenType::SQUARE_C>>(1);
                if (end < 0)
                    throw SyntaxError(tarr[0], "Missing closing ']'");
                return parseLeafMods(new AstIdx(pleft, { tarr, 1, end }), { tarr, end + 1 });
            case TokenType::DOT:
                // member access
                if (tarr.size() < 1 || tarr[1]->ty != TokenType::IDN)
                    throw SyntaxError(tarr[0], "Expected identifier after '.'");
                return parseLeafMods(new AstDot(pleft, tarr[1]), { tarr, 2 });
            }
        }

        template<int prec>
        AstExpr* parseExpr(const TokenArray& tarr);

        // specializations
        template<> AstExpr* parseExpr<3>(const TokenArray& tarr);
        template<> AstExpr* parseExpr<4>(const TokenArray& tarr);
        template<> AstExpr* parseExpr<5>(const TokenArray& tarr);

        template<int prec>
        AstExpr* parseExpr<prec>(const TokenArray& tarr)
        {
            size_t tsz = tarr.size();
            assert(tsz);

            // shortcut for bracketed expressions
            if (tarr[0]->ty == TokenType::ROUND_O && tarr[tsz - 1]->ty == TokenType::ROUND_C)
                return parseExpr<0>(TokenArray(tarr, 1, tsz - 1));
            
            int bbrac = 0;
            int sbrac = 0;
            int abrac = 0;
            for (int i = tsz - 1; i >= 0; i--)
            {
                // Checking for brackets
                switch (tarr[i]->ty)
                {
                case TokenType::ROUND_C:
                    bbrac++;
                    continue;
                case TokenType::ROUND_O:
                    bbrac--;
                    continue;
                case TokenType::SQUARE_C:
                    sbrac++;
                    continue;
                case TokenType::SQUARE_O:
                    sbrac--;
                    continue;
                case TokenType::ANGLE_C:
                    abrac++;
                    continue;
                case TokenType::ANGLE_O:
                    abrac--;
                    continue;
                }

                if (bbrac || sbrac || abrac)
                    continue;

                if (opPrec(tarr[i]) == prec)
                {
                    if (i == tsz - 1)
                        throw SyntaxError(tarr[i], "Missing right side for binary operator");
                    return splitExpr<prec>(tarr, i);
                }
            }
            if (bbrac < 0)
                throw SyntaxError(tarr[0], "Missing closing ')' in expression");
            if (sbrac < 0)
                throw SyntaxError(tarr[0], "Missing closing ']' in expression");
            if (abrac < 0)
                throw SyntaxError(tarr[0], "Missing closing '>' in expression");

            return parseExpr<prec + 1>(tarr);
        }

        // special parsing because of the negative operator
        template<>
        AstExpr* parseExpr<3>(const TokenArray& tarr)
        {
            constexpr int prec = 3;

            size_t tsz = tarr.size();
            assert(tsz);

            // shortcut for bracketed expressions
            if (tarr[0]->ty == TokenType::ROUND_O && tarr[tsz - 1]->ty == TokenType::ROUND_C)
                return parseExpr<0>(TokenArray(tarr, 1, tsz - 1));

            int bbrac = 0;
            int sbrac = 0;
            int abrac = 0;
            for (int i = tsz - 1; i >= 0; i--)
            {
                // Checking for brackets
                switch (tarr[i]->ty)
                {
                case TokenType::ROUND_C:
                    bbrac++;
                    continue;
                case TokenType::ROUND_O:
                    bbrac--;
                    continue;
                case TokenType::SQUARE_C:
                    sbrac++;
                    continue;
                case TokenType::SQUARE_O:
                    sbrac--;
                    continue;
                case TokenType::ANGLE_C:
                    abrac++;
                    continue;
                case TokenType::ANGLE_O:
                    abrac--;
                    continue;
                }

                if (bbrac || sbrac || abrac)
                    continue;

                if (opPrec(tarr[i]) == prec)
                {
                    if (i == 0 && tarr[i]->ty == TokenType::SUB)
                        continue;  // Leaving the negation operator to be parsed with leaf nodes
                    if (i == tsz - 1)
                        throw SyntaxError(tarr[i], "Missing right side for binary operator");
                    return splitExpr<prec>(tarr, i);
                }
            }
            if (bbrac < 0)
                throw SyntaxError(tarr[0], "Missing closing ')' in expression");
            if (sbrac < 0)
                throw SyntaxError(tarr[0], "Missing closing ']' in expression");
            if (abrac < 0)
                throw SyntaxError(tarr[0], "Missing closing '>' in expression");

            return parseExpr<prec + 1>(tarr);
        }

        // special parsing because of the packing operator
        template<>
        AstExpr* parseExpr<4>(const TokenArray& tarr)
        {
            constexpr int prec = 4;

            size_t tsz = tarr.size();
            assert(tsz);

            // shortcut for bracketed expressions
            if (tarr[0]->ty == TokenType::ROUND_O && tarr[tsz - 1]->ty == TokenType::ROUND_C)
                return parseExpr<0>(TokenArray(tarr, 1, tsz - 1));

            int bbrac = 0;
            int sbrac = 0;
            int abrac = 0;
            for (int i = tsz - 1; i >= 0; i--)
            {
                // Checking for brackets
                switch (tarr[i]->ty)
                {
                case TokenType::ROUND_C:
                    bbrac++;
                    continue;
                case TokenType::ROUND_O:
                    bbrac--;
                    continue;
                case TokenType::SQUARE_C:
                    sbrac++;
                    continue;
                case TokenType::SQUARE_O:
                    sbrac--;
                    continue;
                case TokenType::ANGLE_C:
                    abrac++;
                    continue;
                case TokenType::ANGLE_O:
                    abrac--;
                    continue;
                }

                if (bbrac || sbrac || abrac)
                    continue;

                if (opPrec(tarr[i]) == prec)
                {
                    if (i == 0 && tarr[i]->ty == TokenType::STAR)
                        continue;  // Leaving the packing operator to be parsed with leaf nodes
                    if (i == tsz - 1)
                        throw SyntaxError(tarr[i], "Missing right side for binary operator");
                    return splitExpr<prec>(tarr, i);
                }
            }
            if (bbrac < 0)
                throw SyntaxError(tarr[0], "Missing closing ')' in expression");
            if (sbrac < 0)
                throw SyntaxError(tarr[0], "Missing closing ']' in expression");
            if (abrac < 0)
                throw SyntaxError(tarr[0], "Missing closing '>' in expression");

            return parseExpr<prec + 1>(tarr);
        }

        // special parsing for leaf nodes, parseExpr<6> does not exist
        template<>
        AstExpr* parseExpr<5>(const TokenArray& tarr)
        {
            // TODO: leaf node expression parsing ; negative operator, indexing, and literal expressions
            assert(tarr.size() > 0);

            if (tarr.size() == 1)
            {
                // single token leaf node
                switch (tarr[0]->ty)
                {
                case TokenType::INT:
                    return new AstInt(tarr[0]);
                case TokenType::FLOAT:
                    return new AstFloat(tarr[0]);
                case TokenType::STRLIT:
                    return new AstStr(tarr[0]);
                case TokenType::IDN:
                    if (static_cast<const TokenImp<TokenType::IDN>*>(tarr[0])->val == "true")
                        return new AstBool(tarr[0], true);
                    if (static_cast<const TokenImp<TokenType::IDN>*>(tarr[0])->val == "false")
                        return new AstBool(tarr[0], false);
                    return new AstIdn(tarr[0]);
                default:
                    throw SyntaxError(tarr[0], "Unexpected token for single token expression leaf node");
                }
            }
            else
            {
                // multi token leaf node
                switch (tarr[0]->ty)
                {
                case TokenType::SUB:
                    return new AstNeg({ tarr, 1 });
                case TokenType::STAR:
                    return new AstPack({ tarr, 1 });
                case TokenType::ROUND_O:
                {
                    int end = tarr.search<TokenArray::brac_end<TokenType::ROUND_O, TokenType::ROUND_C>>(1);
                    if (end < 0)
                        throw SyntaxError(tarr[0], "Missing closing ')'");
                    return parseLeafMods(parseExpr<0>({ tarr, 1, end }), { tarr, end + 1 });
                }
                case TokenType::IDN:
                    if (static_cast<const TokenImp<TokenType::IDN>*>(tarr[0])->val == "true")
                        throw SyntaxError(tarr[0], "Unexpected token");
                    if (static_cast<const TokenImp<TokenType::IDN>*>(tarr[0])->val == "false")
                        throw SyntaxError(tarr[0], "Unexpected token");
                    return parseLeafMods(new AstIdn(tarr[0]), { tarr, 1 });
                default:
                    throw SyntaxError(tarr[0], "Unexpected start token for multi token expression leaf node");
                }
            }
        }

        // helper for function signatures ie. def my_func<...>(...)
        void paseDefArgs(const TokenArray& tarr, std::vector<AstDecl>& args)
        {
            int start = 0;
            int end;
            do
            {
                end = tarr.search<TokenArray::args_elem<TokenType::ANGLE_O, TokenType::ANGLE_C>>(start);
                if (end == -1)
                    end = tarr.size();
                TokenArray decl(tarr, start, end);
                args.push_back({ decl });
                start = end;
            } while (end != tarr.size());
        }

        // helper for function signatures ie. def my_func<...>(...)
        void parseDefCargs(const TokenArray& tarr, std::vector<AstCargsDecl*>& cargs)
        {
            if (tarr.size() == 0)
                return;

            int start = 0;
            int end;
            do
            {
                end = tarr.search<TokenArray::args_elem<TokenType::ANGLE_O, TokenType::ANGLE_C>>(start);
                if (end == -1)
                    end = tarr.size();
                TokenArray carg(tarr, start, end);
                if (carg.size() == 0)
                    throw SyntaxError(tarr[0], "Dangling end of cargs in def");
                if (carg[0]->ty == TokenType::ANGLE_O)
                {
                    if (carg[carg.size() - 1]->ty != TokenType::ANGLE_C)
                        throw SyntaxError(carg[0], "Missing closing '>' for constarg tuple in def");
                    TokenArray cargtuple(carg, 1, -1);
                    cargs.push_back(new AstCargsTuple(carg));
                }
                else
                    cargs.push_back(new AstDecl(carg));
                start = end + 1;
            } while (end != tarr.size());
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
                end = tarr.search<TokenArray::args_elem<TokenType::ANGLE_O, TokenType::ANGLE_C>>(start);
                if (end == -1)
                    end = tarr.size();
                TokenArray carg(tarr, start, end);
                if (carg.size() == 0)
                    throw SyntaxError(tarr[0], "Dangling end of cargs in block");
                if (carg[0]->ty == TokenType::ANGLE_O)
                {
                    if (carg[carg.size() - 1]->ty != TokenType::ANGLE_C)
                        throw SyntaxError(carg[0], "Missing closing '>' for constarg tuple in block");
                    TokenArray cargtuple(carg, 1, -1);
                    exprs.push_back(new AstTuple(cargtuple));
                }
                else
                    exprs.push_back(parseExpr<0>(carg));
                start = end + 1;
            } while (end != tarr.size());
        }

        // Ast Nodes

        AstBinOp::~AstBinOp()
        {
            delete pleft;
            delete pright;
        }

        AstBool::AstBool(const Token* ptk, bool val)
        {
            assert(ptk->ty == TokenType::IDN);

            line_num = ptk->line_num;
            col_num = ptk->col_num;
            val = val;
        }

        AstInt::AstInt(const Token* ptk)
        {
            assert(ptk->ty == TokenType::INT);

            line_num = ptk->line_num;
            col_num = ptk->col_num;
            val = static_cast<const TokenImp<TokenType::INT>*>(ptk)->val;
        }

        AstFloat::AstFloat(const Token* ptk)
        {
            assert(ptk->ty == TokenType::FLOAT);

            line_num = ptk->line_num;
            col_num = ptk->col_num;
            val = static_cast<const TokenImp<TokenType::FLOAT>*>(ptk)->val;
        }

        AstStr::AstStr(const Token* ptk)
        {
            assert(ptk->ty == TokenType::STRLIT);

            line_num = ptk->line_num;
            col_num = ptk->col_num;
            val = static_cast<const TokenImp<TokenType::STRLIT>*>(ptk)->val;
        }

        AstIdn::AstIdn(const Token* ptk)
        {
            assert(ptk->ty == TokenType::IDN);

            line_num = ptk->line_num;
            col_num = ptk->col_num;
            idn = static_cast<const TokenImp<TokenType::IDN>*>(ptk)->val;
        }

        AstTuple::AstTuple(const TokenArray& tarr)
        {
            assert(tarr.size() != 0);
            line_num = tarr[0]->line_num;
            col_num = tarr[0]->col_num;

            int start = 0;
            int end;
            do
            {
                end = tarr.search<TokenArray::is_same_brac<TokenType::COMMA>>(start);
                if (end == -1)
                    end = tarr.size();
                if (end == start)
                    throw SyntaxError(tarr[start], "Empty vararg parameter");
                elems.push_back(parseExpr<1>({ tarr, start, end }));
                start = end + 1;
                if (start == tarr.size())
                    throw SyntaxError(tarr[end], "Empty vararg parameter");
            } while (end != tarr.size());
        }

        AstCall::AstCall(AstExpr* pleft, const TokenArray& tarr)
        {
            assert(tarr.size() != 0);
            line_num = tarr[0]->line_num;
            col_num = tarr[0]->col_num;

            this->pleft = pleft;

            int start = 0;
            int end;
            do
            {
                end = tarr.search<TokenArray::is_same_brac<TokenType::COMMA>>(start);
                if (end == -1)
                    end = tarr.size();
                if (end == start)
                    throw SyntaxError(tarr[start], "Empty vararg parameter");
                args.push_back(parseExpr<1>({ tarr, start, end }));
                start = end + 1;
                if (start == tarr.size())
                    throw SyntaxError(tarr[end], "Empty vararg parameter");
            } while (end != tarr.size());
        }

        AstCargs::AstCargs(AstExpr* pleft, const TokenArray& tarr)
        {
            assert(tarr.size() != 0);
            line_num = tarr[0]->line_num;
            col_num = tarr[0]->col_num;

            this->pleft = pleft;

            int start = 0;
            int end;
            do
            {
                end = tarr.search<TokenArray::is_same_brac<TokenType::COMMA>>(start);
                if (end == -1)
                    end = tarr.size();
                if (end == start)
                    throw SyntaxError(tarr[start], "Empty constarg parameter");
                args.push_back(parseExpr<1>({ tarr, start, end }));
                start = end + 1;
                if (start == tarr.size())
                    throw SyntaxError(tarr[end], "Empty constarg parameter");
            } while (end != tarr.size());
        }

        AstIdx::AstIdx(AstExpr* pleft, const TokenArray& tarr)
        {
            assert(tarr.size() != 0);
            line_num = tarr[0]->line_num;
            col_num = tarr[0]->col_num;

            this->pleft = pleft;

            int start = 0;
            int end;
            do
            {
                end = tarr.search<TokenArray::is_same_brac<TokenType::COMMA>>(start);
                if (end == -1)
                    end = tarr.size();
                if (end == start)
                    throw SyntaxError(tarr[start], "Empty index parameter");  // [..., , ...]
                parseSlice({ tarr, start, end });
                start = end + 1;
                if (start == tarr.size())
                    throw SyntaxError(tarr[end], "Empty index parameter");  // [...,]
            } while (end != tarr.size());
        }

        void AstIdx::parseSlice(const TokenArray& tarr)
        {
            assert(tarr.size() != 0);

            std::vector<AstExpr*> slices;
            int start = 0;
            int end;
            do
            {
                end = tarr.search<TokenArray::is_same_brac<TokenType::COLON>>(start);
                if (end == -1)
                    end = tarr.size();
                TokenArray tslice(tarr, start, end);
                if (tslice.size() == 0)
                    slices.push_back(NULL);  // [...::...]
                else
                    slices.push_back(parseExpr<1>(tslice));  // [...:...:...]

                start = end + 1;
                if (start == tarr.size())  // [...:]
                {
                    slices.push_back(NULL);
                    indicies.push_back(slices);
                    return;
                }
            } while (end != tarr.size());
        }

        AstDot::AstDot(AstExpr* pleft, const Token* ptk)
        {
            assert(ptk->ty == TokenType::IDN);

            line_num = ptk->line_num;
            col_num = ptk->col_num;
            this->pleft = pleft;
            this->member = static_cast<const TokenImp<TokenType::IDN>*>(ptk)->val;
        }

        AstNeg::AstNeg(const TokenArray& tarr)
        {
            assert(tarr.size() != 0);
            line_num = tarr[0]->line_num;
            col_num = tarr[0]->col_num;

            pexpr = parseExpr<5>(tarr);
        }

        AstPack::AstPack(const TokenArray& tarr)
        {
            assert(tarr.size() != 0);
            line_num = tarr[0]->line_num;
            col_num = tarr[0]->col_num;

            pexpr = parseExpr<5>(tarr);
        }

        AstAdd::AstAdd(const TokenArray& left, const TokenArray& right)
        {
            assert(left.size() != 0);
            line_num = left[0]->line_num;
            col_num = left[0]->col_num;

            pleft = parseExpr<4>(left);
            pright = parseExpr<3>(right);
        }

        AstSub::AstSub(const TokenArray& left, const TokenArray& right)
        {
            assert(left.size() != 0);
            line_num = left[0]->line_num;
            col_num = left[0]->col_num;

            pleft = parseExpr<4>(left);
            pright = parseExpr<3>(right);
        }

        AstMul::AstMul(const TokenArray& left, const TokenArray& right)
        {
            assert(left.size() != 0);
            line_num = left[0]->line_num;
            col_num = left[0]->col_num;

            pleft = parseExpr<5>(left);
            pright = parseExpr<4>(right);
        }

        AstDiv::AstDiv(const TokenArray& left, const TokenArray& right)
        {
            assert(left.size() != 0);
            line_num = left[0]->line_num;
            col_num = left[0]->col_num;

            pleft = parseExpr<5>(left);
            pright = parseExpr<4>(right);
        }

        AstEq::AstEq(const TokenArray& left, const TokenArray& right)
        {
            assert(left.size() != 0);
            line_num = left[0]->line_num;
            col_num = left[0]->col_num;

            pleft = parseExpr<3>(left);
            pright = parseExpr<2>(right);
        }

        AstNe::AstNe(const TokenArray& left, const TokenArray& right)
        {
            assert(left.size() != 0);
            line_num = left[0]->line_num;
            col_num = left[0]->col_num;

            pleft = parseExpr<3>(left);
            pright = parseExpr<2>(right);
        }

        AstGe::AstGe(const TokenArray& left, const TokenArray& right)
        {
            assert(left.size() != 0);
            line_num = left[0]->line_num;
            col_num = left[0]->col_num;

            pleft = parseExpr<3>(left);
            pright = parseExpr<2>(right);
        }

        AstLe::AstLe(const TokenArray& left, const TokenArray& right)
        {
            assert(left.size() != 0);
            line_num = left[0]->line_num;
            col_num = left[0]->col_num;

            pleft = parseExpr<3>(left);
            pright = parseExpr<2>(right);
        }

        AstGt::AstGt(const TokenArray& left, const TokenArray& right)
        {
            assert(left.size() != 0);
            line_num = left[0]->line_num;
            col_num = left[0]->col_num;

            pleft = parseExpr<3>(left);
            pright = parseExpr<2>(right);
        }

        AstLt::AstLt(const TokenArray& left, const TokenArray& right)
        {
            assert(left.size() != 0);
            line_num = left[0]->line_num;
            col_num = left[0]->col_num;

            pleft = parseExpr<3>(left);
            pright = parseExpr<2>(right);
        }

        AstAnd::AstAnd(const TokenArray& left, const TokenArray& right)
        {
            assert(left.size() != 0);
            line_num = left[0]->line_num;
            col_num = left[0]->col_num;

            pleft = parseExpr<2>(left);
            pright = parseExpr<1>(right);
        }

        AstOr::AstOr(const TokenArray& left, const TokenArray& right)
        {
            assert(left.size() != 0);
            line_num = left[0]->line_num;
            col_num = left[0]->col_num;

            pleft = parseExpr<2>(left);
            pright = parseExpr<1>(right);
        }

        AstIAdd::AstIAdd(const TokenArray& left, const TokenArray& right)
        {
            assert(left.size() != 0);
            line_num = left[0]->line_num;
            col_num = left[0]->col_num;

            pleft = parseExpr<1>(left);
            pright = parseExpr<0>(right);
        }
        
        AstISub::AstISub(const TokenArray& left, const TokenArray& right)
        {
            assert(left.size() != 0);
            line_num = left[0]->line_num;
            col_num = left[0]->col_num;

            pleft = parseExpr<1>(left);
            pright = parseExpr<0>(right);
        }

        AstIMul::AstIMul(const TokenArray& left, const TokenArray& right)
        {
            assert(left.size() != 0);
            line_num = left[0]->line_num;
            col_num = left[0]->col_num;

            pleft = parseExpr<1>(left);
            pright = parseExpr<0>(right);
        }
        
        AstIDiv::AstIDiv(const TokenArray& left, const TokenArray& right)
        {
            assert(left.size() != 0);
            line_num = left[0]->line_num;
            col_num = left[0]->col_num;

            pleft = parseExpr<1>(left);
            pright = parseExpr<0>(right);
        }

        AstAssign::AstAssign(const TokenArray& left, const TokenArray& right)
        {
            assert(left.size() != 0);
            line_num = left[0]->line_num;
            col_num = left[0]->col_num;

            decl_assign = isDecl(left);
            if (decl_assign)
                pleft = new AstDecl(left);
            else
                pleft = parseExpr<1>(left);
            pright = parseExpr<0>(right);
        }

        AstDecl::AstDecl()
        {
            this->line_num = 0;
            this->col_num = 0;
            this->var_name = "";
            this->type_name = "";
            this->cargs = {};
        }

        AstDecl::AstDecl(const TokenArray& tarr)
        {
            assert(tarr.size());
            if (tarr.size() < 2)
                throw SyntaxError(tarr[0], "Invalid syntax for declaration");
            line_num = tarr[0]->line_num;
            col_num = tarr[0]->col_num;

            size_t start = 0;
            is_static = false;
            if (tarr[0]->ty == TokenType::IDN && static_cast<const TokenImp<TokenType::IDN>*>(tarr[0])->val == "static")
            {
                is_static = true;
                start++;
            }

            if (tarr[start]->ty != TokenType::IDN)
                throw SyntaxError(tarr[start], "Invalid token for type name in declaration");
            this->type_name = static_cast<const TokenImp<TokenType::IDN>*>(tarr[start])->val;
            if (this->type_name == "var")
            {
                if (is_static)
                    throw SyntaxError(tarr[start], "static var declarations are not allowed");
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
                
                if (tarr.size() != start + 2)
                {
                    if (tarr[start + 2]->ty != TokenType::ANGLE_O)
                        throw SyntaxError(tarr[start + 2], "Expected '<' in type for variable declaration");
                    if (tarr[tarr.size() - 2]->ty != TokenType::ANGLE_C)
                        throw SyntaxError(tarr[tarr.size() - 2], "Expected '>' in type for variable declaration");
                    TokenArray cargs(tarr, start + 3, tarr.size() - 2);
                    parseConstargs(cargs, this->cargs);
                }
            }
        }

        AstDecl::~AstDecl()
        {
            for (auto e : cargs)
                delete e;
        }

        AstIf::AstIf(const TokenArray& if_sig, const TokenArray& if_seq, int indent_level) :
            seq(if_seq, indent_level + 1)
        {
            assert(if_sig.size() != 0);
            assert(if_sig[0]->ty == TokenType::IDN);
            assert(static_cast<const TokenImp<TokenType::IDN>*>(if_sig[0])->val == "if");
            line_num = if_sig[0]->line_num;
            col_num = if_sig[0]->col_num;

            if (if_sig.size() < 2)  // if cond  <- minimum number of tokens in an if statement signature
                throw SyntaxError(if_sig[0], "Invalid if statement signature");

            this->pcond = parseExpr<0>(TokenArray(if_sig, 1));
        }

        AstIf::~AstIf()
        {
            if (pcond)
                delete pcond;
        }

        AstWhile::AstWhile(const TokenArray& while_sig, const TokenArray& while_seq, int indent_level) :
            seq(while_seq, indent_level + 1)
        {
            assert(while_sig.size() != 0);
            assert(while_sig[0]->ty == TokenType::IDN);
            assert(static_cast<const TokenImp<TokenType::IDN>*>(while_sig[0])->val == "while");
            line_num = while_sig[0]->line_num;
            col_num = while_sig[0]->col_num;

            if (while_sig.size() < 2)  // while cond  <- minimum number of tokens in a while loop signature
                throw SyntaxError(while_sig[0], "Invalid while loop signature");

            this->pcond = parseExpr<0>(TokenArray(while_sig, 1));
        }

        AstWhile::~AstWhile()
        {
            if (pcond)
                delete pcond;
        }

        AstFor::AstFor(const TokenArray& for_sig, const TokenArray& for_seq, int indent_level) :
            seq(for_seq, indent_level + 1)
        {
            assert(for_sig.size() != 0);
            assert(for_sig[0]->ty == TokenType::IDN);
            assert(static_cast<const TokenImp<TokenType::IDN>*>(for_sig[0])->val == "for");
            line_num = for_sig[0]->line_num;
            col_num = for_sig[0]->col_num;

            if (for_sig.size() < 5)  // for var i in lst  <- minimum number of tokens in a for loop signature
                throw SyntaxError(for_sig[0], "Invalid for loop signature");

            int in_pos = for_sig.search<TokenArray::is_keyword<Keyword::IN>>(2);
            if (in_pos < 0)
                throw SyntaxError(for_sig[1], "Missing keyword in for loop signature: 'in'");
            this->it = TokenArray(for_sig, 1, in_pos);  // decl init
            this->pexpr = parseExpr<0>(TokenArray(for_sig, in_pos + 1));
        }

        AstFor::~AstFor()
        {
            if (pexpr)
                delete pexpr;
        }

        AstCargsTuple::AstCargsTuple(const TokenArray& tarr)
        {
            parseDefCargs(tarr, this->elems);
        }

        AstCargsTuple::~AstCargsTuple()
        {
            for (auto e : elems)
                delete e;
        }

        AstSeq::AstSeq(const TokenArray& tarr, int indent_level)
        {
            assert(tarr.size() != 0);
            line_num = tarr[0]->line_num;
            col_num = tarr[0]->col_num;

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

                end = tarr.search<TokenArray::is_same<TokenType::ENDL>>(start);
                if (end < 0)
                    end = tarr.size();
                TokenArray expr_tarr(tarr, start, end);
                this->blocks.push_back(parseExpr<0>(expr_tarr));

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
            assert(def_sig.size() != 0);
            line_num = def_sig[0]->line_num;
            col_num = def_sig[0]->col_num;

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
                end = def_sig.search<TokenArray::brac_end<TokenType::ANGLE_O, TokenType::ANGLE_C>>();
                if (end < 0)
                    throw SyntaxError(def_sig[start], "Missing closing '>' in function signature");
                TokenArray constargs_tarr({ def_sig, start, end });
                parseDefCargs(constargs_tarr, cargs);
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
            paseDefArgs(varargs_tarr, this->vargs);
        }

        AstModImp::AstModImp(const TokenArray& tarr)
        {
            assert(tarr.size() != 0);
            line_num = tarr[0]->line_num;
            col_num = tarr[0]->col_num;

            if (tarr[0]->ty != TokenType::IDN)
                throw SyntaxError(tarr[0], "Expected identifier");
            imp.push_back(static_cast<const TokenImp<TokenType::IDN>*>(tarr[0])->val);

            for (int i = 1; i < tarr.size(); i++)
            {
                if (tarr[i]->ty != TokenType::DOT)
                    throw SyntaxError(tarr[i], "Expected '.'");
                
                i++;
                if (i == tarr.size())
                    throw SyntaxError(tarr[i - 1], "Expected token after '.'");
                if (tarr[i]->ty == TokenType::STAR)
                    imp.push_back("");  // empty string means *
                if (tarr[i]->ty != TokenType::IDN)
                    imp.push_back(static_cast<const TokenImp<TokenType::IDN>*>(tarr[i])->val);
                throw SyntaxError(tarr[i - 1], "Expected identifier or '*' after '.'"); 
            }
        }

        AstModule::AstModule(const TokenArray& tarr)
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
                    i++;
                    int end = tarr.search<TokenArray::is_same<TokenType::ENDL>>(i);
                    if (end == i)
                        throw SyntaxError(tarr[i], "Expected identifier after 'import'");
                    this->imps.push_back(AstModImp({ tarr, i, end }));
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
    }
}
