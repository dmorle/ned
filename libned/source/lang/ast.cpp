#include <ned/errors.h>
#include <ned/lang/ast.h>
#include <ned/lang/lexer.h>
#include <ned/lang/obj.h>
#include <ned/lang/bytecode.h>

#include <functional>
#include <cassert>

#include <iostream>

namespace nn
{
    namespace lang
    {
        std::string to_string(ExprKW kw)
        {
            switch (kw)
            {
            case ExprKW::TYPE:
                return "type";
            case ExprKW::INIT:
                return "type";
            case ExprKW::FTY:
                return "type";
            case ExprKW::BOOL:
                return "type";
            case ExprKW::INT:
                return "type";
            case ExprKW::FLOAT:
                return "type";
            case ExprKW::STR:
                return "type";
            case ExprKW::ARRAY:
                return "type";
            case ExprKW::TUPLE:
                return "type";
            case ExprKW::F16:
                return "type";
            case ExprKW::F32:
                return "type";
            case ExprKW::F64:
                return "type";
            default:
                return "INVALID";
            }
        }

        // Parses comma deliminated token sequences within a matching set of brackets
        // Returns the index in tarr immediately after the CLOSE token was found
        // On failure, this function returns -1
        template<typename T, bool(*parse_fn)(const TokenArray&, T&), TokenType OPEN, TokenType CLOSE>
        inline int parse_args(const TokenArray& tarr, int i, std::vector<T>& args)
        {
            assert(tarr[i]->ty == OPEN);

            // Advancing one token past the opening bracket
            i++;
            int end = tarr.search(ArgEndCriteria(CLOSE), i);
            if (end == -1)
                return error::syntax(tarr[0], "Unable to find the closing token %", to_string(CLOSE));
            if (tarr[end]->ty == CLOSE)
            {
                // either a single arg, or its empty
                for (int j = i; j < end; j++)
                    if (!tarr[j]->is_whitespace())
                    {
                        // single arg, parse it and move on
                        args.push_back(T());
                        if (parse_fn({ tarr, i, end }, args.back()))
                        {
                            args.clear();
                            return -1;
                        }
                        break;
                    }
                return end + 1;
            }

            // General case for parsing multiple args
            args.push_back(T());
            if (parse_fn({ tarr, i, end }, args.back()))
            {
                args.clear();
                return -1;
            }
            i = end + 1;
            do
            {
                end = tarr.search(ArgEndCriteria(CLOSE), i);
                args.push_back(T());
                if (parse_fn({ tarr, i, end }, args.back()))
                {
                    args.clear();
                    return -1;
                }
                i = end + 1;
            } while (tarr[end]->ty == TokenType::COMMA);

            return i;
        }

        // Parses lines from a token sequence
        // On failure, this function returns true
        template<typename T, bool(*parse_fn)(const TokenArray&, T&, int)>
        inline bool parse_lines(const TokenArray& tarr, int i, int indent_level, std::vector<T>& lines)
        {
            int end;
            do
            {
                end = tarr.search(LineEndCriteria(indent_level), i);
                if (end < 0)
                    end = tarr.size();
                for (int j = i; j < end; j++)  // Only use lines that have non-whitespace tokens in them
                    if (!tarr[j]->is_whitespace())
                    {
                        lines.push_back(T());
                        if (parse_fn({ tarr, i, end }, lines.back(), indent_level))
                        {
                            lines.clear();
                            return true;
                        }
                        break;
                    }
                i = end + 1;
            } while (end < tarr.size());
            return false;
        }

        template<>
        bool parse_signature<AstCargSig>(const TokenArray& tarr, AstCargSig& ast_carg_sig)
        {
            assert(tarr.size() > 0);

            // Setup and getting the name of the definition
            int i = 0;
            for (; tarr[i]->is_whitespace(); i++);
            if (tarr[i]->expect<TokenType::IDN>())
                return true;
            ast_carg_sig.name = tarr[i]->get<TokenType::IDN>().val;
            for (i++; i < tarr.size() && tarr[i]->is_whitespace(); i++);
            if (i == tarr.size())
                return false;

            // Parsing out the cargs themselves
            if (tarr[i]->expect<TokenType::ANGLE_O>())
                return true;
            i = parse_args<AstArgDecl, parse_arg_decl, TokenType::ANGLE_O, TokenType::ANGLE_C>(tarr, i, ast_carg_sig.cargs);
            if (i == -1)
                return error::syntax(tarr[0], "Error parsing cargs");
            for (; i < tarr.size() && tarr[i]->is_whitespace(); i++);
            if (i != tarr.size())
                return error::syntax(tarr[i], "Unexpected token after cargs");
            return false;
        }

        template<>
        bool parse_signature<AstFnSig>(const TokenArray& tarr, AstFnSig& ast_fn_sig)
        {
            assert(tarr.size() > 0);

            // Setup and getting the name of the fn definition
            int i = 0;
            for (; tarr[i]->is_whitespace(); i++);
            if (tarr[i]->expect<TokenType::IDN>())
                return true;
            ast_fn_sig.name = tarr[i]->get<TokenType::IDN>().val;
            for (i++; i < tarr.size() && tarr[i]->is_whitespace(); i++);
            if (i == tarr.size())
                return error::syntax(tarr[tarr.size() - 1], "Unexpected end of signature after name");

            // Parsing out the cargs from the fn definition
            if (tarr[i]->ty == TokenType::ANGLE_O)
            {
                i = parse_args<AstArgDecl, parse_arg_decl, TokenType::ANGLE_O, TokenType::ANGLE_C>(tarr, i, ast_fn_sig.cargs);
                if (i == -1)
                    return true;
                for (; i < tarr.size() && tarr[i]->is_whitespace(); i++);
                if (i == tarr.size())
                    return error::syntax(tarr[tarr.size() - 1], "Unexpected end of signature after cargs");
            }

            // Parsing out the vargs from the fn definition
            if (tarr[i]->expect<TokenType::ROUND_O>())
                return true;
            i = parse_args<AstArgDecl, parse_arg_decl, TokenType::ROUND_O, TokenType::ROUND_C>(tarr, i, ast_fn_sig.vargs);
            if (i == -1)
                return true;
            for (; i < tarr.size() && tarr[i]->is_whitespace(); i++);
            if (i == tarr.size())
                return false;  // No return values

            // Parsing out the rets from the fn definition
            if (tarr[i]->expect<TokenType::ARROW>())
                return true;
            for (i++; i < tarr.size() && tarr[i]->is_whitespace(); i++);
            if (i == tarr.size())
                return error::syntax(tarr[tarr.size() - 1], "Expected an expression after '->'");
            // commas are valid operators in expression parsing resulting in a tuple
            // so I can reuse the expression parsing I already wrote to parse all the rets in the signature
            // then if a tuple is returned, I can just flatten it.  Otherwise, theres a single ret.
            // This has the added value of allowing for bracketed return values, which is needed for function references
            AstExpr ret_expr;
            if (parse_expr({ tarr, i }, ret_expr))
                return true;
            if (ret_expr.ty == ExprType::LIT_TUPLE)
                ast_fn_sig.rets = std::move(ret_expr.expr_agg.elems);
            else
                ast_fn_sig.rets.push_back(std::move(ret_expr));

            return false;
        }

        template<>
        bool parse_signature<AstBlockSig>(const TokenArray& tarr, AstBlockSig& ast_block_sig)
        {
            assert(tarr.size() > 0);

            // Setup and getting the name of the block definition
            int i = 0;
            for (; tarr[i]->is_whitespace(); i++);
            if (tarr[i]->expect<TokenType::IDN>())
                return true;
            ast_block_sig.name = tarr[i]->get<TokenType::IDN>().val;
            for (i++; i < tarr.size() && tarr[i]->is_whitespace(); i++);
            if (i == tarr.size())
                return error::syntax(tarr[tarr.size() - 1], "Unexpected end of signature after name");

            // Parsing out the cargs from the block definition
            if (tarr[i]->ty == TokenType::ANGLE_O)
            {
                i = parse_args<AstArgDecl, parse_arg_decl, TokenType::ANGLE_O, TokenType::ANGLE_C>(tarr, i, ast_block_sig.cargs);
                if (i == -1)
                    return true;
                for (; i < tarr.size() && tarr[i]->is_whitespace(); i++);
                if (i == tarr.size())
                    return error::syntax(tarr[tarr.size() - 1], "Unexpected end of signature after cargs");
            }

            // Parsing out the vargs from the block definition
            if (tarr[i]->expect<TokenType::ROUND_O>())
                return true;
            i = parse_args<AstArgDecl, parse_arg_decl, TokenType::ROUND_O, TokenType::ROUND_C>(tarr, i, ast_block_sig.vargs);
            if (i == -1)
                return true;
            for (; i < tarr.size() && tarr[i]->is_whitespace(); i++);
            if (i == tarr.size())
                return false;  // No return values

            // Parsing out the first ret from the block definition
            if (tarr[i]->expect<TokenType::ARROW>())
                return true;
            for (i++; i < tarr.size() && tarr[i]->is_whitespace(); i++);
            if (i == tarr.size())
                return error::syntax(tarr[tarr.size() - 1], "Expected an indentifier after '->'");
            int end = tarr.size();
            if (tarr[i]->ty == TokenType::ROUND_O)  // Allowing for bracketed rets (mainly for block references)
            {
                end = tarr.search(IsSameCriteria(TokenType::ROUND_C), ++i);
                if (end == -1)
                    return error::syntax(tarr[i], "Missing closing ')' for block return values");
                // making sure everything past the ')' is whitespace
                for (const Token& tk : TokenArray(tarr, end))
                    if (!tk.is_whitespace())
                        return error::syntax(tk, "Unexpected token after block return values");
                for (i++; i < end && tarr[i]->is_whitespace(); i++);  // catching up start
                if (i == end)  // no return values: "-> ()"
                    return false;
            }
            if (tarr[i]->expect<TokenType::IDN>())
                return true;
            ast_block_sig.rets.push_back(tarr[i]->get<TokenType::IDN>().val);
            for (i++; i < tarr.size() && tarr[i]->is_whitespace(); i++);

            // Parsing out all the other rets
            while (i < end)
            {
                if (tarr[i]->expect<TokenType::COMMA>())
                    return true;
                for (i++; i < end && tarr[i]->is_whitespace(); i++);
                if (i == end)
                    return error::syntax(tarr[(size_t)i - 1], "Expected identifier after ',' in block return values");
                if (tarr[i]->expect<TokenType::IDN>())
                    return true;
                ast_block_sig.rets.push_back(tarr[i]->get<TokenType::IDN>().val);
                for (i++; i < end && tarr[i]->is_whitespace(); i++);
            }
            return false;
        }

        enum class OpAssoc
        {
            LTOR,
            RTOL
        };

        // Eventually this can become constexpr, but it seems my current c++ compiler doesn't have that yet...
        const std::vector<std::pair<OpAssoc, std::vector<TokenType>>> prec_ops = {
            { OpAssoc::RTOL, { TokenType::ASSIGN, TokenType::IADD, TokenType::ISUB, TokenType::IMUL, TokenType::IDIV, TokenType::IMOD, TokenType::IPOW }},
            { OpAssoc::LTOR, { TokenType::COMMA }},
            { OpAssoc::LTOR, { TokenType::KW_AND, TokenType::KW_OR }},
            { OpAssoc::LTOR, { TokenType::CMP_EQ, TokenType::CMP_NE, TokenType::CMP_GT, TokenType::CMP_LT, TokenType::CMP_GE, TokenType::CMP_LE }},
            { OpAssoc::LTOR, { TokenType::ADD, TokenType::SUB }},
            { OpAssoc::LTOR, { TokenType::STAR, TokenType::DIV, TokenType::MOD }},
            { OpAssoc::LTOR, { TokenType::POW } },
            { OpAssoc::RTOL, { TokenType::CAST } }
        };

        bool parse_index(const TokenArray& tarr, std::vector<AstExprIndex::Elem>& elems)
        {
            assert(tarr.size() > 0);
            int start = 0, end;
            while (true)
            {
                for (; start < tarr.size() && tarr[start]->is_whitespace(); start++);
                if (start >= tarr.size())
                    break;
                if (tarr[start]->ty == TokenType::ELLIPSES)
                {
                    // tarr[start] should be the only white space character between start and
                    // the next comma up to end
                    for (end = start + 1; end < tarr.size() && tarr[end]->is_whitespace(); end++);
                    if (end == tarr.size())
                    {
                        // The index expression ends in "..."
                        elems.push_back({ AstExprIndex::Elem::Type::ELLIPSES });
                        return false;
                    }
                    if (tarr[end]->expect<TokenType::COMMA>())
                        return true;
                    start = end + 1;  // one token past the comma
                    continue;
                }

                // non-ellipses elems
                end = tarr.search(IsSameCriteria(TokenType::COMMA), start);
                if (end == -1)
                    end = tarr.size();
                else if (end == start)
                    return error::syntax(tarr[start], "Found empty index element expression");

                int mid = tarr.search(IsSameCriteria(TokenType::COLON), start);
                AstExprIndex::Elem elem;
                if (mid == -1)
                {
                    // Non-sliced index
                    elem.ty = AstExprIndex::Elem::Type::DIRECT;
                    elem.lhs = std::make_unique<AstExpr>();
                    if (parse_expr({ tarr, start, end }, *elem.lhs))
                        return true;
                }
                else
                {
                    // Sliced index
                    elem.ty = AstExprIndex::Elem::Type::SLICE;
                    if (start != mid)
                    {
                        // A lower index was found.  If start was equal to mid, it means that the first
                        // whitespace token in the index element was a colon, which is valid syntax
                        elem.lhs = std::make_unique<AstExpr>();
                        if (parse_expr({ tarr, start, mid }, *elem.lhs))
                            return true;
                    }
                    for (mid++; mid < end && tarr[mid]->is_whitespace(); mid++);
                    if (mid != end)
                    {
                        // Similar logic to the if (start != mid) thing.  Even in the case where
                        // both the lhs and rhs are null, its still valid syntax
                        elem.rhs = std::make_unique<AstExpr>();
                        if (parse_expr({ tarr, mid, end }, *elem.rhs))
                            return true;
                    }
                }

                start = end + 1;
                elems.push_back(std::move(elem));
            }

            if (elems.size() == 0)
                return error::syntax(tarr[0], "Found empty index expression");
            return false;
        }

        bool parse_leaf_mods(const TokenArray& tarr, AstExpr& ast_expr, std::unique_ptr<AstExpr> lhs)
        {
            assert(tarr.size() && !tarr[0]->is_whitespace());

            // Finding the bounds on the sub expression
            int start, end, i;  // start of subexpr, end of subexpr, beginning of possible next expr
            switch (tarr[0]->ty)
            {
            case TokenType::SQUARE_O:
                for (start = 1; start < tarr.size() && tarr[start]->is_whitespace(); start++);
                end = tarr.search(IsSameCriteria(TokenType::SQUARE_C));  // It needs to see the '[' otherwise it'll ignore the ']'
                if (end == -1)
                    return error::syntax(tarr[0], "Missing closing ']' in index expression");
                i = end + 1;
                break;
            case TokenType::DOT:
                for (start = 1; start < tarr.size() && tarr[start]->is_whitespace(); start++);
                end = tarr.search(IsSameCriteria(TokenType::IDN), start);
                if (end == -1)
                    end = tarr.size();
                else
                    end++;  // i needs to go one past the identifier
                for (const Token& tk : TokenArray{ tarr, start, end - 1 })  // making sure there are no more tokens between the identifier and the dot
                    if (!tk.is_whitespace())
                        return error::syntax(tk, "Unexpected token after '.'");
                i = end;
                break;
            case TokenType::IDN:
                for (const Token& tk : TokenArray{ tarr, 1 })  // making sure there are no more tokens
                    if (!tk.is_whitespace())
                        return error::syntax(tk, "Unexpected token after variable declaration");
                new (&ast_expr.expr_name) AstExprName();
                ast_expr.ty = ExprType::VAR_DECL;
                ast_expr.expr_name.expr = std::move(lhs);
                ast_expr.expr_name.val = tarr[0]->get<TokenType::IDN>().val;
                return false;
            case TokenType::ANGLE_O:
                for (start = 1; start < tarr.size() && tarr[start]->is_whitespace(); start++);
                end = tarr.search(IsSameCriteria(TokenType::ANGLE_C));
                if (end == -1)
                    return error::syntax(tarr[0], "Missing closing '>' in carg expression");
                i = end + 1;
                break;
            case TokenType::ROUND_O:
                for (start = 1; start < tarr.size() && tarr[start]->is_whitespace(); start++);
                end = tarr.search(IsSameCriteria(TokenType::ROUND_C));
                if (end == -1)
                    return error::syntax(tarr[0], "Missing closing ')' in varg expression");
                i = end + 1;
                break;
            default:
                return error::syntax(tarr[0], "Unexpected token in leafmod expression");
            }

            std::unique_ptr<AstExpr> nlhs;
            AstExpr* pexpr;
            if (i == tarr.size())
                pexpr = &ast_expr;
            else
            {
                pexpr = (nlhs = std::make_unique<AstExpr>()).get();
                pexpr->node_info = {
                    .fname = tarr[0]->fname,
                    .line_start = tarr[0]->line_num,
                    .line_end = tarr[tarr.size() - 1]->line_num,
                    .col_start = tarr[0]->col_num,
                    .col_end = tarr[tarr.size() - 1]->col_num
                };
            }

            if (end == start)
            {
                // It could be an empty carg or varg call
                AstExpr ret_expr;
                switch (tarr[0]->ty)
                {
                case TokenType::ANGLE_O:
                    new (&pexpr->expr_call) AstExprCall();
                    pexpr->ty = ExprType::CARGS_CALL;
                    pexpr->expr_call.callee = std::move(lhs);
                    break;
                case TokenType::ROUND_O:
                    new (&pexpr->expr_call) AstExprCall();
                    pexpr->ty = ExprType::VARGS_CALL;
                    pexpr->expr_call.callee = std::move(lhs);
                    break;
                default:
                    return error::syntax(tarr[start], "Empty expression");
                }
                if (i == tarr.size())
                    return false;
                return parse_leaf_mods({ tarr, i }, ast_expr, std::move(nlhs));
            }

            AstExpr ret_expr;  // helper for calls so that I leverge the tuple parsing code for this
            switch (tarr[0]->ty)
            {
            case TokenType::SQUARE_O:
                // TODO: implement index with slicing
                new (&pexpr->expr_binary) AstExprIndex();
                pexpr->ty = ExprType::INDEX;
                pexpr->expr_index.expr = std::move(lhs);
                if (parse_index({ tarr, start, end }, pexpr->expr_index.args))
                    return true;
                break;
            case TokenType::DOT:
                switch (tarr[start]->ty)
                {
                case TokenType::IDN:
                    new (&pexpr->expr_name) AstExprName();
                    pexpr->ty = ExprType::DOT;
                    pexpr->expr_name.expr = std::move(lhs);
                    pexpr->expr_name.val = tarr[start]->get<TokenType::IDN>().val;
                    break;
                case TokenType::KW_FORWARD:
                    new (&pexpr->expr_unary) AstExprUnaryOp();
                    pexpr->ty = ExprType::UNARY_FORWARD;
                    pexpr->expr_unary.expr = std::move(lhs);
                    break;
                case TokenType::KW_BACKWARD:
                    new (&pexpr->expr_unary) AstExprUnaryOp();
                    pexpr->ty = ExprType::UNARY_BACKWARD;
                    pexpr->expr_unary.expr = std::move(lhs);
                    break;
                default:
                    return error::syntax(tarr[start], "Unexpected token after '.'");
                }
                break;
            case TokenType::ANGLE_O:
                new (&pexpr->expr_call) AstExprCall();
                pexpr->ty = ExprType::CARGS_CALL;
                pexpr->expr_call.callee = std::move(lhs);
                if (parse_expr({ tarr, start, end }, ret_expr))
                    return true;
                if (ret_expr.ty == ExprType::LIT_TUPLE)
                    pexpr->expr_call.args = std::move(ret_expr.expr_agg.elems);
                else
                    pexpr->expr_call.args.push_back(std::move(ret_expr));
                break;
            case TokenType::ROUND_O:
                new (&pexpr->expr_call) AstExprCall();
                pexpr->ty = ExprType::VARGS_CALL;
                pexpr->expr_call.callee = std::move(lhs);
                if (parse_expr({ tarr, start, end }, ret_expr))
                    return true;
                if (ret_expr.ty == ExprType::LIT_TUPLE)
                    pexpr->expr_call.args = std::move(ret_expr.expr_agg.elems);
                else
                    pexpr->expr_call.args.push_back(std::move(ret_expr));
                break;
            default:
                assert(false);
            }

            if (i == tarr.size())
                return false;
            return parse_leaf_mods({ tarr, i }, ast_expr, std::move(nlhs));
        }

        bool parse_leaf_token(const Token* ptk, AstExpr& ast_expr)
        {
            switch (ptk->ty)
            {
            case TokenType::LIT_INT:
                ast_expr.expr_int = ptk->get<TokenType::LIT_INT>().val;
                ast_expr.ty = ExprType::LIT_INT;
                return false;
            case TokenType::LIT_FLOAT:
                ast_expr.expr_float = ptk->get<TokenType::LIT_FLOAT>().val;
                ast_expr.ty = ExprType::LIT_FLOAT;
                return false;
            case TokenType::LIT_STR:
                new (&ast_expr.expr_string) std::string(ptk->get<TokenType::LIT_STR>().val);
                ast_expr.ty = ExprType::LIT_STRING;
                return false;
            case TokenType::IDN:
                new (&ast_expr.expr_string) std::string(ptk->get<TokenType::IDN>().val);
                ast_expr.ty = ExprType::VAR;
                return false;
            case TokenType::KW_TRUE:
                ast_expr.expr_bool = true;
                ast_expr.ty = ExprType::LIT_BOOL;
                return false;
            case TokenType::KW_FALSE:
                ast_expr.expr_bool = false;
                ast_expr.ty = ExprType::LIT_BOOL;
                return false;
            case TokenType::KW_NULL:
                ast_expr.expr_kw = ExprKW::NUL;
                ast_expr.ty = ExprType::KW;
                return false;
            case TokenType::KW_TYPE:
                ast_expr.expr_kw = ExprKW::TYPE;
                ast_expr.ty = ExprType::KW;
                return false;
            case TokenType::KW_VOID:
                ast_expr.expr_kw = ExprKW::VOID;
                ast_expr.ty = ExprType::KW;
                return false;
            case TokenType::KW_INIT:
                ast_expr.expr_kw = ExprKW::INIT;
                ast_expr.ty = ExprType::KW;
                return false;
            case TokenType::KW_FTY:
                ast_expr.expr_kw = ExprKW::FTY;
                ast_expr.ty = ExprType::KW;
                return false;
            case TokenType::KW_BOOL:
                ast_expr.expr_kw = ExprKW::BOOL;
                ast_expr.ty = ExprType::KW;
                return false;
            case TokenType::KW_INT:
                ast_expr.expr_kw = ExprKW::INT;
                ast_expr.ty = ExprType::KW;
                return false;
            case TokenType::KW_FLOAT:
                ast_expr.expr_kw = ExprKW::FLOAT;
                ast_expr.ty = ExprType::KW;
                return false;
            case TokenType::KW_STR:
                ast_expr.expr_kw = ExprKW::STR;
                ast_expr.ty = ExprType::KW;
                return false;
            case TokenType::KW_ARRAY:
                ast_expr.expr_kw = ExprKW::ARRAY;
                ast_expr.ty = ExprType::KW;
                return false;
            case TokenType::KW_TUPLE:
                ast_expr.expr_kw = ExprKW::TUPLE;
                ast_expr.ty = ExprType::KW;
                return false;
            case TokenType::KW_CFG:
                ast_expr.expr_kw = ExprKW::CFG;
                ast_expr.ty = ExprType::KW;
                return false;
            case TokenType::KW_F16:
                ast_expr.expr_kw = ExprKW::F16;
                ast_expr.ty = ExprType::KW;
                return false;
            case TokenType::KW_F32:
                ast_expr.expr_kw = ExprKW::F32;
                ast_expr.ty = ExprType::KW;
                return false;
            case TokenType::KW_F64:
                ast_expr.expr_kw = ExprKW::F64;
                ast_expr.ty = ExprType::KW;
                return false;
            default:
                error::syntax(ptk, "Unexpected token for single token expression leaf node");
                return true;
            }
        }

        template<int prec>
        inline ExprType get_expr_type(TokenType ty);

        template<>
        inline ExprType get_expr_type<0>(TokenType ty)
        {
            switch (ty)
            {
            case TokenType::ASSIGN:
                return ExprType::BINARY_ASSIGN;
            case TokenType::IADD:
                return ExprType::BINARY_IADD;
            case TokenType::ISUB:
                return ExprType::BINARY_ISUB;
            case TokenType::IMUL:
                return ExprType::BINARY_IMUL;
            case TokenType::IDIV:
                return ExprType::BINARY_IDIV;
            case TokenType::IMOD:
                return ExprType::BINARY_IMOD;
            case TokenType::IPOW:
                return ExprType::BINARY_IPOW;
            }
            assert(false);
            return ExprType::INVALID;
        }

        template<>
        inline ExprType get_expr_type<2>(TokenType ty)
        {
            switch (ty)
            {
            case TokenType::KW_AND:
                return ExprType::BINARY_AND;
            case TokenType::KW_OR:
                return ExprType::BINARY_OR;
            }
            assert(false);
            return ExprType::INVALID;
        }

        template<>
        inline ExprType get_expr_type<3>(TokenType ty)
        {
            switch (ty)
            {
            case TokenType::CMP_EQ:
                return ExprType::BINARY_CMP_EQ;
            case TokenType::CMP_NE:
                return ExprType::BINARY_CMP_NE;
            case TokenType::CMP_GT:
                return ExprType::BINARY_CMP_GT;
            case TokenType::CMP_LT:
                return ExprType::BINARY_CMP_LT;
            case TokenType::CMP_GE:
                return ExprType::BINARY_CMP_GE;
            case TokenType::CMP_LE:
                return ExprType::BINARY_CMP_LE;
            }
            assert(false);
            return ExprType::INVALID;
        }

        template<>
        inline ExprType get_expr_type<4>(TokenType ty)
        {
            switch (ty)
            {
            case TokenType::ADD:
                return ExprType::BINARY_ADD;
            case TokenType::SUB:
                return ExprType::BINARY_SUB;
            }
            assert(false);
            return ExprType::INVALID;
        }

        template<>
        inline ExprType get_expr_type<5>(TokenType ty)
        {
            switch (ty)
            {
            case TokenType::STAR:
                return ExprType::BINARY_MUL;
            case TokenType::DIV:
                return ExprType::BINARY_DIV;
            case TokenType::MOD:
                return ExprType::BINARY_MOD;
            }
            assert(false);
            return ExprType::INVALID;
        }

        template<>
        inline ExprType get_expr_type<6>(TokenType ty)
        {
            switch (ty)
            {
            case TokenType::POW:
                return ExprType::BINARY_POW;
            }
            assert(false);
            return ExprType::INVALID;
        }

        template<>
        inline ExprType get_expr_type<7>(TokenType ty)
        {
            switch (ty)
            {
            case TokenType::CAST:
                return ExprType::BINARY_CAST;
            }
            assert(false);
            return ExprType::INVALID;
        }

        template<int prec>
        bool parse_expr_prec(const TokenArray& tarr, AstExpr& ast_expr);

        // specializations
        template<> bool parse_expr_prec<9>(const TokenArray& tarr, AstExpr& ast_expr);
        template<> bool parse_expr_prec<8>(const TokenArray& tarr, AstExpr& ast_expr);
        template<> bool parse_expr_prec<1>(const TokenArray& tarr, AstExpr& ast_expr);

        template<int prec>
        bool parse_expr_prec(const TokenArray& tarr, AstExpr& ast_expr)
        {
            static_assert(0 <= prec && prec <= 7 && prec != 1);
            assert(tarr.size() > 0);
            // Note to self: never trust the MSVC compiler to not fuck up structured bindings...
            OpAssoc op_assoc = std::get<0>(prec_ops[prec]);
            std::vector<TokenType> tys = std::get<1>(prec_ops[prec]);
            int pos;
            if (op_assoc == OpAssoc::LTOR)
                pos = tarr.rsearch(IsInCriteria(tys));
            else
                pos = tarr.search(IsInCriteria(tys));
            if (pos < 1)  // allow for pos == 0 cause of unary operators
                return parse_expr_prec<prec + 1>(tarr, ast_expr);
            if (tarr.size() - pos < 2)
                return error::syntax(tarr[pos], "Dangling operator");
            // I need to explicitly initialize the node_info for pleft and pright
            // since they don't go into parse_expr, they won't get initialized automatically.
            std::unique_ptr<AstExpr> pleft = std::make_unique<AstExpr>();
            pleft->node_info = {
                .fname = tarr[0]->fname,
                .line_start = tarr[0]->line_num,
                .line_end = tarr[pos]->line_num,
                .col_start = tarr[0]->col_num,
                .col_end = tarr[pos]->col_num
            };
            std::unique_ptr<AstExpr> pright = std::make_unique<AstExpr>();
            pright->node_info = {
                .fname = tarr[0]->fname,
                .line_start = tarr[pos + 1]->line_num,
                .line_end = tarr[tarr.size() - 1]->line_num,
                .col_start = tarr[pos + 1]->col_num,
                .col_end = tarr[tarr.size() - 1]->col_num
            };
            if (op_assoc == OpAssoc::LTOR)
            {
                if (parse_expr_prec<prec>({ tarr, 0, pos }, *pleft) ||
                    parse_expr_prec<prec + 1>({ tarr, pos + 1 }, *pright)
                    ) return true;
            }
            else
            {
                if (parse_expr_prec<prec + 1>({ tarr, 0, pos }, *pleft) ||
                    parse_expr_prec<prec>({ tarr, pos + 1 }, *pright)
                    ) return true;
            }
            new (&ast_expr.expr_binary) AstExprBinaryOp();
            ast_expr.ty = get_expr_type<prec>(tarr[pos]->ty);
            ast_expr.expr_binary.left = std::move(pleft);
            ast_expr.expr_binary.right = std::move(pright);
            return false;
        }

        template<>
        bool parse_expr_prec<9>(const TokenArray& tarr, AstExpr& ast_expr)
        {
            assert(tarr.size() > 0);

            if (tarr.size() == 1)  // single token leaf node
                return parse_leaf_token(tarr[0], ast_expr);

            std::unique_ptr<AstExpr> lhs = nullptr;
            int i;
            if (tarr[0]->ty == TokenType::ROUND_O)
            {
                // handling bracketed expressions
                i = tarr.search(IsSameCriteria(TokenType::ROUND_C));
                if (i == -1)
                    return error::syntax(tarr[0], "Missing closing ')'");
                if (i == 1)
                    return error::syntax(tarr[0], "Empty expression are not allowed");

                // Checking if all the remaining tokens are whitespace
                // This is used to determine if any mods exist on the expression
                int end;
                for (end = i + 1; end < tarr.size() && tarr[end]->is_whitespace(); end++);
                if (end == tarr.size())
                    return parse_expr({ tarr, 1, i }, ast_expr);

                lhs = std::make_unique<AstExpr>();
                if (parse_expr({ tarr, 1, i }, *lhs))
                    return true;
                i = end;
            }
            else if (tarr[0]->ty == TokenType::SQUARE_O)
            {
                // handling array literals
                i = tarr.search(IsSameCriteria(TokenType::SQUARE_C));
                if (i == -1)
                    return error::syntax(tarr[0], "Missing closing ']'");

                // Checking if all the remaining tokens are whitespace
                // This is used to determine if any mods exist on the array literal
                int end;
                for (end = i + 1; end < tarr.size() && tarr[end]->is_whitespace(); end++);

                // Checking if the array is empty
                bool is_empty = true;
                for (int j = 1; j < i; j++)
                    if (!tarr[j]->is_whitespace())
                    {
                        is_empty = false;
                        break;
                    }

                if (is_empty)
                {
                    // Empty array
                    if (end != tarr.size())
                    {
                        // has mods => initialize lhs and continue execution
                        lhs = std::make_unique<AstExpr>();
                        lhs->node_info = {
                            .fname = tarr[0]->fname,
                            .line_start = tarr[0]->line_num,
                            .line_end = tarr[tarr.size() - 1]->line_num,
                            .col_start = tarr[0]->col_num,
                            .col_end = tarr[tarr.size() - 1]->col_num
                        };
                        new (&lhs->expr_agg) AstExprAggLit();
                        lhs->ty = ExprType::LIT_ARRAY;
                        i = end;
                    }
                    else
                    {
                        // does not have mods => initialize ast_expr and return
                        new (&ast_expr.expr_agg) AstExprAggLit();
                        ast_expr.ty = ExprType::LIT_ARRAY;
                        return false;
                    }
                }
                else
                {
                    // Non-empty array
                    AstExpr arr_expr;
                    if (parse_expr({ tarr, 1, i }, arr_expr))
                        return true;
                    if (end != tarr.size())
                    {
                        // has mods => initialize lhs and continue execution
                        lhs = std::make_unique<AstExpr>();
                        lhs->node_info = {
                            .fname = tarr[0]->fname,
                            .line_start = tarr[0]->line_num,
                            .line_end = tarr[tarr.size() - 1]->line_num,
                            .col_start = tarr[0]->col_num,
                            .col_end = tarr[tarr.size() - 1]->col_num
                        };
                        new (&lhs->expr_agg) AstExprAggLit();
                        lhs->ty = ExprType::LIT_ARRAY;
                        if (arr_expr.ty == ExprType::LIT_TUPLE)
                            lhs->expr_agg.elems = std::move(arr_expr.expr_agg.elems);
                        else
                            lhs->expr_agg.elems.push_back(std::move(arr_expr));
                        i = end;
                    }
                    else
                    {
                        // does not have mods => initialize ast_expr and return
                        new (&ast_expr.expr_agg) AstExprAggLit();
                        ast_expr.ty = ExprType::LIT_ARRAY;
                        if (arr_expr.ty == ExprType::LIT_TUPLE)
                            ast_expr.expr_agg.elems = std::move(arr_expr.expr_agg.elems);
                        else
                            ast_expr.expr_agg.elems.push_back(std::move(arr_expr));
                        return false;
                    }
                }
            }
            else if (tarr[0]->ty == TokenType::KW_DEF)
            {
                // Block reference declaration
                new (&ast_expr.expr_blk_decl) AstBlockSig();
                ast_expr.ty = ExprType::DEF_DECL;
                return parse_signature(tarr, ast_expr.expr_blk_decl);
            }
            else if (tarr[0]->ty == TokenType::KW_INTR)
            {
                // Intrinsic reference declaration
                new (&ast_expr.expr_blk_decl) AstBlockSig();
                ast_expr.ty = ExprType::INTR_DECL;
                return parse_signature(tarr, ast_expr.expr_blk_decl);
            }
            else if (tarr[0]->ty == TokenType::KW_FN)
            {
                // Function reference declaration
                new (&ast_expr.expr_fn_decl) AstFnSig();
                ast_expr.ty = ExprType::FN_DECL;
                return parse_signature(tarr, ast_expr.expr_fn_decl);
            }
            else
            {
                // single token followed by modifiers
                lhs = std::make_unique<AstExpr>();
                lhs->node_info = {
                    .fname = tarr[0]->fname,
                    .line_start = tarr[0]->line_num,
                    .line_end = tarr[0]->line_num,
                    .col_start = tarr[0]->col_num,
                    .col_end = tarr[0]->col_num
                };
                if (parse_leaf_token(tarr[0], *lhs))
                    return true;
                i = 1;
            }

            // If execution falls through the above code,
            // it means lhs and start are initialized and there are leaf mods that need to be parsed
            // start should be the index immediately after the lhs expression
            return parse_leaf_mods({ tarr, i }, ast_expr, std::move(lhs));
        }

        template<>
        bool parse_expr_prec<8>(const TokenArray& tarr, AstExpr& ast_expr)
        {
            assert(tarr.size() > 0);
            constexpr int prec = 8;

            // Handling unary operators
            std::unique_ptr<AstExpr> pexpr = nullptr;
            switch (tarr[0]->ty)
            {
                // Don't move repeated code out.  In the case where the type falls through,
                // keeping the code inside each case will save logic and a malloc
            case TokenType::ADD:
                if (tarr.size() < 2)
                    return error::syntax(tarr[0], "Dangling operator");
                pexpr = std::make_unique<AstExpr>();

                if (parse_expr_prec<prec + 1>({ tarr, 1 }, *pexpr))
                    return true;
                new (&ast_expr.expr_unary) AstExprUnaryOp();
                ast_expr.ty = ExprType::UNARY_POS;
                ast_expr.expr_unary.expr = std::move(pexpr);
                return false;
            case TokenType::SUB:
                if (tarr.size() < 2)
                    return error::syntax(tarr[0], "Dangling operator");
                pexpr = std::make_unique<AstExpr>();
                pexpr->node_info = {
                    .fname = tarr[0]->fname,
                    .line_start = tarr[1]->line_num,
                    .line_end = tarr[tarr.size() - 1]->line_num,
                    .col_start = tarr[1]->col_num,
                    .col_end = tarr[tarr.size() - 1]->col_num
                };
                if (parse_expr_prec<prec + 1>({ tarr, 1 }, *pexpr))
                    return true;
                new (&ast_expr.expr_unary) AstExprUnaryOp();
                ast_expr.ty = ExprType::UNARY_NEG;
                ast_expr.expr_unary.expr = std::move(pexpr);
                return false;
            case TokenType::STAR:
                if (tarr.size() < 2)
                    return error::syntax(tarr[0], "Dangling operator");
                pexpr = std::make_unique<AstExpr>();
                pexpr->node_info = {
                    .fname = tarr[0]->fname,
                    .line_start = tarr[1]->line_num,
                    .line_end = tarr[tarr.size() - 1]->line_num,
                    .col_start = tarr[1]->col_num,
                    .col_end = tarr[tarr.size() - 1]->col_num
                };
                if (parse_expr_prec<prec + 1>({ tarr, 1 }, *pexpr))
                    return true;
                new (&ast_expr.expr_unary) AstExprUnaryOp();
                ast_expr.ty = ExprType::UNARY_UNPACK;
                ast_expr.expr_unary.expr = std::move(pexpr);
                return false;
            case TokenType::KW_NOT:
                if (tarr.size() < 2)
                    return error::syntax(tarr[0], "Dangling operator");
                pexpr = std::make_unique<AstExpr>();
                pexpr->node_info = {
                    .fname = tarr[0]->fname,
                    .line_start = tarr[1]->line_num,
                    .line_end = tarr[tarr.size() - 1]->line_num,
                    .col_start = tarr[1]->col_num,
                    .col_end = tarr[tarr.size() - 1]->col_num
                };
                if (parse_expr_prec<prec + 1>({ tarr, 1 }, *pexpr))
                    return true;
                new (&ast_expr.expr_unary) AstExprUnaryOp();
                ast_expr.ty = ExprType::UNARY_NOT;
                ast_expr.expr_unary.expr = std::move(pexpr);
                return false;
            case TokenType::KW_REF:
                if (tarr.size() < 2)
                    return error::syntax(tarr[0], "Dangling operator");
                pexpr = std::make_unique<AstExpr>();
                pexpr->node_info = {
                    .fname = tarr[0]->fname,
                    .line_start = tarr[1]->line_num,
                    .line_end = tarr[tarr.size() - 1]->line_num,
                    .col_start = tarr[1]->col_num,
                    .col_end = tarr[tarr.size() - 1]->col_num
                };
                if (parse_expr_prec<prec + 1>({ tarr, 1 }, *pexpr))
                    return true;
                new (&ast_expr.expr_unary) AstExprUnaryOp();
                ast_expr.ty = ExprType::UNARY_REF;
                ast_expr.expr_unary.expr = std::move(pexpr);
                return false;
            case TokenType::KW_CONST:
                if (tarr.size() < 2)
                    return error::syntax(tarr[0], "Dangling operator");
                pexpr = std::make_unique<AstExpr>();
                pexpr->node_info = {
                    .fname = tarr[0]->fname,
                    .line_start = tarr[1]->line_num,
                    .line_end = tarr[tarr.size() - 1]->line_num,
                    .col_start = tarr[1]->col_num,
                    .col_end = tarr[tarr.size() - 1]->col_num
                };
                if (parse_expr_prec<prec + 1>({ tarr, 1 }, *pexpr))
                    return true;
                new (&ast_expr.expr_unary) AstExprUnaryOp();
                ast_expr.ty = ExprType::UNARY_CONST;
                ast_expr.expr_unary.expr = std::move(pexpr);
                return false;
            }

            return parse_expr_prec<prec + 1>(tarr, ast_expr);
        }

        template<>
        bool parse_expr_prec<1>(const TokenArray& tarr, AstExpr& ast_expr)
        {
            assert(tarr.size() > 0);
            constexpr int prec = 1;

            // Handling the comma operator
            IsSameCriteria sc{ TokenType::COMMA };
            int i = 0;
            int end;
            std::vector<AstExpr> elem_exprs;
            while ((end = tarr.search(sc, i)) > 0)
            {
                AstExpr elem_expr;
                if (parse_expr({ tarr, i, end }, elem_expr))
                    return true;
                elem_exprs.push_back(std::move(elem_expr));
                i = end + 1;
            }

            if (i == 0)  // No commas were found
                return parse_expr_prec<prec + 1>(tarr, ast_expr);

            new (&ast_expr.expr_agg) AstExprAggLit();
            ast_expr.ty = ExprType::LIT_TUPLE;
            ast_expr.expr_agg.elems = std::move(elem_exprs);
            if (i == tarr.size())
                return false;  // In case the tuple was ended with a comma
            AstExpr last_expr;
            if (parse_expr({ tarr, i }, last_expr))
                return true;
            ast_expr.expr_agg.elems.push_back(std::move(last_expr));
            return false;
        }

        bool parse_expr(const TokenArray& tarr, AstExpr& ast_expr)
        {
            assert(tarr.size() > 0);
            ast_expr.node_info = {
                .fname = tarr[0]->fname,
                .line_start = tarr[0]->line_num,
                .line_end = tarr[tarr.size() - 1]->line_num,
                .col_start = tarr[0]->col_num,
                .col_end = tarr[tarr.size() - 1]->col_num
            };
            
            int i = 0;
            for (; i < tarr.size() && tarr[i]->is_whitespace(); i++);
            if (i == tarr.size())
                return error::syntax(tarr[0], "Empty expression are not allowed");
            int end = tarr.size() - 1;
            for (; tarr[end]->is_whitespace(); end--);
            return parse_expr_prec<0>({ tarr, i, end + 1 }, ast_expr);
        }

        bool parse_match_elem(const TokenArray& tarr, AstMatchElem& ast_match_elem, int indent_level)
        {
            assert(tarr.size() > 0);
            ast_match_elem.node_info = {
                .fname = tarr[0]->fname,
                .line_start = tarr[0]->line_num,
                .line_end = tarr[tarr.size() - 1]->line_num,
                .col_start = tarr[0]->col_num,
                .col_end = tarr[tarr.size() - 1]->col_num
            };

            int i = 0;
            for (; i < tarr.size() && tarr[i]->is_whitespace(); i++);
            if (i == tarr.size())
                return error::syntax(tarr[0], "Empty lines are not allowed");

            // Parsing out the label of the match statement element
            if (tarr[i]->ty == TokenType::KW_ELSE)
                ast_match_elem.label = "else";  // Let the compiler figure this out
            else
            {
                if (tarr[i]->expect<TokenType::IDN>())
                    return true;
                ast_match_elem.label = tarr[i]->get<TokenType::IDN>().val;
            }
            
            // Parsing out the body of the match statement element
            int end = i + 1;
            for (; end < tarr.size() && tarr[end]->is_whitespace(); end++);
            if (end == tarr.size())
                return error::syntax(tarr[i], "Missing ':' after label % in match statement", ast_match_elem.label);
            if (tarr[end]->expect<TokenType::COLON>())
                return true;
            for (end++; end < tarr.size() && tarr[end]->ty == TokenType::INDENT; end++);
            if (end == tarr.size())
                return error::syntax(tarr[(size_t)end - 1], "Missing body of label % in match statement", ast_match_elem.label);
            return
                tarr[end]->expect<TokenType::ENDL>() ||
                parse_lines<AstLine, parse_line>(tarr, end, indent_level + 1, ast_match_elem.body);
        }

        bool parse_line(const TokenArray& tarr, AstLine& ast_line, int indent_level)
        {
            assert(tarr.size() > 0);
            ast_line.node_info = {
                .fname = tarr[0]->fname,
                .line_start = tarr[0]->line_num,
                .line_end = tarr[tarr.size() - 1]->line_num,
                .col_start = tarr[0]->col_num,
                .col_end = tarr[tarr.size() - 1]->col_num
            };

            int i = 0;
            for (; i < tarr.size() && tarr[i]->is_whitespace(); i++);
            if (i == tarr.size())
                return error::syntax(tarr[0], "Empty lines are not allowed");
            int end;
            switch (tarr[i]->ty)
            {
            case TokenType::KW_BREAK:
                for (const auto& tk : TokenArray{ tarr, 1 })
                    if (!tk.is_whitespace())
                        return error::syntax(tarr[0], "Unexpected token in break statement");
                ast_line.ty = LineType::BREAK;
                break;

            case TokenType::KW_CONTINUE:
                for (const auto& tk : TokenArray{ tarr, 1 })
                    if (!tk.is_whitespace())
                        return error::syntax(tarr[0], "Unexpected token in continue statement");
                ast_line.ty = LineType::CONTINUE;
                break;

            case TokenType::KW_EXPORT:
                for (i++; tarr[i]->is_whitespace(); i++);
                if (tarr[i]->expect<TokenType::IDN>())
                    return true;
                new (&ast_line.line_export) AstLineExport();
                ast_line.ty = LineType::EXPORT;
                ast_line.line_export.var_name = tarr[i]->get<TokenType::IDN>().val;
                break;

            case TokenType::KW_EXTERN:
                for (end = tarr.size() - 1; end > 0 && tarr[end]->is_whitespace(); end--);
                if (end == 0)
                    return error::syntax(tarr[0], "Expected init specification and variable after 'extern'");
                if (tarr[end]->expect<TokenType::IDN>())
                    return true;
                new (&ast_line.line_extern) AstLineExtern();
                ast_line.ty = LineType::EXTERN;
                ast_line.line_extern.var_name = tarr[end]->get<TokenType::IDN>().val;
                if (parse_expr({ tarr, i + 1, end }, ast_line.line_extern.init_expr))
                    return error::syntax(tarr[1], "Invalid syntax in weight init expression");
                break;

            case TokenType::KW_RETURN:
                new (&ast_line.line_func) AstLineUnaryFunc();
                ast_line.ty = LineType::RETURN;
                if (parse_expr({ tarr, i + 1 }, ast_line.line_func.expr))
                    return error::syntax(tarr[i], "Invalid syntax in return statement");
                break;

            case TokenType::KW_ADD_INTR_INFO:
                new (&ast_line.line_intrinfo) AstLineIntrInfo();
                ast_line.ty = LineType::INTRINFO;

                end = tarr.search(IsSameCriteria(TokenType::COLON), i + 1);
                if (end == -1)
                    return error::syntax(tarr[i], "Missing ':' in __add_intr_info statement");
                if (end == i + 1)
                    return error::syntax(tarr[i], "Empty name expression in __add_intr_info statement");
                if (parse_expr({ tarr, i + 1, end }, ast_line.line_intrinfo.name_expr))
                    return error::syntax(tarr[i], "Invalid syntax in __add_intr_info name expression");
                if (end + 1 == tarr.size())
                    return error::syntax(tarr[(size_t)end - 1], "Empty cfg expression in __add_intr_info statement");
                if (parse_expr({ tarr, end + 1 }, ast_line.line_intrinfo.cfg_expr))
                    return error::syntax(tarr[end + 1], "Invalid syntax in __add_intr_info cfg expression");
                return false;

            case TokenType::KW_MATCH:
                new (&ast_line.line_match) AstLineMatch();
                ast_line.ty = LineType::MATCH;
                
                end = tarr.search(IsSameCriteria(TokenType::COLON), i + 1);
                if (end == -1)
                    return error::syntax(tarr[i], "Missing ':' in match statement");
                if (end == i + 1)
                    return error::syntax(tarr[i], "Empty expression in match condition");
                if (parse_expr({ tarr, i + 1, end }, ast_line.line_match.arg))
                    return error::syntax(tarr[i], "Invalid syntax in match argument");

                for (end++; end < tarr.size() && tarr[end]->ty == TokenType::INDENT; end++);
                if (end == tarr.size())
                    return error::syntax(tarr[(size_t)end - 1], "Missing body of match block");
                if (tarr[end]->expect<TokenType::ENDL>() ||
                    parse_lines<AstMatchElem, parse_match_elem>(tarr, end, indent_level + 1, ast_line.line_match.elems))
                    return true;
                break;

            case TokenType::KW_IF:
                new (&ast_line.line_branch) AstLineBranch();
                ast_line.ty = LineType::IF;

                end = tarr.search(IsSameCriteria(TokenType::COLON), i + 1);
                if (end == -1)
                    return error::syntax(tarr[i], "Missing ':' in if statement");
                if (end == i + 1)
                    return error::syntax(tarr[i], "Empty expression in if condition");
                if (parse_expr({ tarr, i + 1, end }, ast_line.line_branch.cond))
                    return error::syntax(tarr[i], "Invalid syntax in if condition");

                for (end++; end < tarr.size() && tarr[end]->ty == TokenType::INDENT; end++);
                if (end == tarr.size())
                    return error::syntax(tarr[(size_t)end - 1], "Missing body of evaluation mode block");
                if (tarr[end]->expect<TokenType::ENDL>() ||
                    parse_lines<AstLine, parse_line>(tarr, end, indent_level + 1, ast_line.line_branch.body))
                    return true;
                break;

            case TokenType::KW_ELIF:
                new (&ast_line.line_branch) AstLineBranch();
                ast_line.ty = LineType::ELIF;

                end = tarr.search(IsSameCriteria(TokenType::COLON), i + 1);
                if (end == -1)
                    return error::syntax(tarr[i], "Missing ':' in elif statement");
                if (end == i + 1)
                    return error::syntax(tarr[i], "Empty expression in elif condition");
                if (parse_expr({ tarr, i + 1, end }, ast_line.line_branch.cond))
                    return error::syntax(tarr[i], "Invalid syntax in elif condition");

                for (end++; end < tarr.size() && tarr[end]->ty == TokenType::INDENT; end++);
                if (end == tarr.size())
                    return error::syntax(tarr[(size_t)end - 1], "Missing body of evaluation mode block");
                if (tarr[end]->expect<TokenType::ENDL>() ||
                    parse_lines<AstLine, parse_line>(tarr, end, indent_level + 1, ast_line.line_branch.body))
                    return true;
                break;

            case TokenType::KW_ELSE:
                new (&ast_line.line_block) AstLineBlock();
                ast_line.ty = LineType::ELSE;
                
                for (end = i + 1; tarr[i]->is_whitespace(); i++);
                if (tarr[end]->expect<TokenType::COLON>())
                    return true;

                for (end++; end < tarr.size() && tarr[end]->ty == TokenType::INDENT; end++);
                if (end == tarr.size())
                    return error::syntax(tarr[(size_t)end - 1], "Missing body of evaluation mode block");
                if (tarr[end]->expect<TokenType::ENDL>() ||
                    parse_lines<AstLine, parse_line>(tarr, end, indent_level + 1, ast_line.line_block.body))
                    return true;
                break;

            case TokenType::KW_WHILE:
                new (&ast_line.line_branch) AstLineBranch();
                ast_line.ty = LineType::WHILE;

                end = tarr.search(IsSameCriteria(TokenType::COLON));
                if (end == -1)
                    return error::syntax(tarr[i], "Missing ':' in while loop");
                if (end == i + 1)
                    return error::syntax(tarr[i], "Empty expression in while loop condition");
                if (parse_expr({ tarr, i + 1, end }, ast_line.line_branch.cond))
                    return error::syntax(tarr[i], "Invalid syntax in while condition");

                for (end++; end < tarr.size() && tarr[end]->ty == TokenType::INDENT; end++);
                if (end == tarr.size())
                    return error::syntax(tarr[(size_t)end - 1], "Missing body of evaluation mode block");
                if (tarr[end]->expect<TokenType::ENDL>() ||
                    parse_lines<AstLine, parse_line>(tarr, end, indent_level + 1, ast_line.line_branch.body))
                    return true;
                break;

            case TokenType::KW_FOR:
                new (&ast_line.line_for) AstLineFor();
                ast_line.ty = LineType::FOR;

                end = tarr.search(IsSameCriteria(TokenType::KW_IN), i);
                if (end == -1)
                    return error::syntax(tarr[i], "Missing 'in' keyword in for loop");
                if (end == i + 1)
                    return error::syntax(tarr[i], "Empty expression in for loop counter");
                if (parse_expr({ tarr, i + 1, end }, ast_line.line_for.decl))
                    return error::syntax(tarr[i], "Invalid syntax in for loop counter");
                i = end;
                
                end = tarr.search(IsSameCriteria(TokenType::COLON), i);
                if (end == -1)
                    return error::syntax(tarr[i], "Missing ':' in for loop");
                if (end == i + 1)
                    return error::syntax(tarr[i], "Empty expression in for loop iterator");
                if (parse_expr({ tarr, i + 1, end }, ast_line.line_for.iter))
                    return error::syntax(tarr[i], "Invalid syntax in for loop iterator");
                
                for (end++; end < tarr.size() && tarr[end]->ty == TokenType::INDENT; end++);
                if (end == tarr.size())
                    return error::syntax(tarr[(size_t)end - 1], "Missing body of evaluation mode block");
                if (tarr[end]->expect<TokenType::ENDL>() ||
                    parse_lines<AstLine, parse_line>(tarr, end, indent_level + 1, ast_line.line_for.body))
                    return true;
                break;

            case TokenType::MODE:
                new (&ast_line.line_label) AstLineLabel();
                ast_line.ty = LineType::EVALMODE;

                for (i++; i < tarr.size() && tarr[i]->is_whitespace(); i++);
                if (tarr[i]->expect<TokenType::IDN>())
                    return true;
                ast_line.line_label.label = tarr[i]->get<TokenType::IDN>().val;
                for (i++; i < tarr.size() && tarr[i]->is_whitespace(); i++);
                if (tarr[i]->expect<TokenType::COLON>())
                    return true;

                for (i++; i < tarr.size() && tarr[i]->ty == TokenType::INDENT; i++);
                if (i == tarr.size())
                    return error::syntax(tarr[(size_t)i - 1], "Missing body of evaluation mode block");
                if (tarr[i]->expect<TokenType::ENDL>() ||
                    parse_lines<AstLine, parse_line>(tarr, i, indent_level + 1, ast_line.line_label.body))
                    return true;
                break;

            default:
                new (&ast_line.line_expr) AstLineExpr();
                ast_line.ty = LineType::EXPR;
                if (parse_expr({ tarr, i }, ast_line.line_expr.line))
                    return true;
                break;
            }
            return false;
        }

        bool parse_arg_decl(const TokenArray& tarr, AstArgDecl& ast_arg_decl)
        {
            assert(tarr.size() > 0);
            ast_arg_decl.node_info = {
                .fname = tarr[0]->fname,
                .line_start = tarr[0]->line_num,
                .line_end = tarr[tarr.size() - 1]->line_num,
                .col_start = tarr[0]->col_num,
                .col_end = tarr[tarr.size() - 1]->col_num
            };

            // handling default values
            int i = tarr.search(IsSameCriteria(TokenType::ASSIGN));
            if (i != -1)
            {
                i++;
                if (i == tarr.size())
                    return error::syntax(tarr[tarr.size() - 1], "Expected expression after '='");
                ast_arg_decl.default_expr = std::make_unique<AstExpr>();
                if (parse_expr({ tarr, i }, *ast_arg_decl.default_expr))
                    return true;
                i--;
            }
            else
            {
                ast_arg_decl.default_expr = nullptr;
                i = tarr.size();
            }

            // handling references to defs/intrs/fns.  The syntax for these are different from normal declarations
            int start;
            for (start = 0; start < i && tarr[start]->is_whitespace(); start++);
            if (start == i)
                return error::syntax(tarr[tarr.size() - 1], "Expected an expression");
            switch (tarr[0]->ty)
            {
            case TokenType::KW_DEF:
                for (start++; start < i && tarr[start]->is_whitespace(); start++);
                if (start == i)
                    return error::syntax(tarr[i], "Expected signature after 'def'");
                if (tarr[start]->ty == TokenType::STAR)
                {
                    ast_arg_decl.is_packed = true;
                    for (start++; start < i && tarr[start]->is_whitespace(); start++);
                    if (start == i)
                        return error::syntax(tarr[i], "Expected signature after 'def'");
                }
                if (tarr[start]->expect<TokenType::IDN>())
                    return true;
                ast_arg_decl.var_name = tarr[start]->get<TokenType::IDN>().val;
                ast_arg_decl.type_expr = std::make_unique<AstExpr>();
                new (&ast_arg_decl.type_expr->expr_blk_decl) AstBlockSig();
                ast_arg_decl.type_expr->ty = ExprType::DEF_DECL;
                return parse_signature(tarr, ast_arg_decl.type_expr->expr_blk_decl);

            case TokenType::KW_INTR:
                for (start++; start < i && tarr[start]->is_whitespace(); start++);
                if (start == i)
                    return error::syntax(tarr[i], "Expected signature after 'intr'");
                if (tarr[start]->ty == TokenType::STAR)
                {
                    ast_arg_decl.is_packed = true;
                    for (start++; start < i && tarr[start]->is_whitespace(); start++);
                    if (start == i)
                        return error::syntax(tarr[i], "Expected signature after 'intr'");
                }
                if (tarr[start]->expect<TokenType::IDN>())
                    return true;
                ast_arg_decl.var_name = tarr[start]->get<TokenType::IDN>().val;
                ast_arg_decl.type_expr = std::make_unique<AstExpr>();
                new (&ast_arg_decl.type_expr->expr_blk_decl) AstBlockSig();
                ast_arg_decl.type_expr->ty = ExprType::INTR_DECL;
                return parse_signature(tarr, ast_arg_decl.type_expr->expr_blk_decl);

            case TokenType::KW_FN:
                for (start++; start < i && tarr[start]->is_whitespace(); start++);
                if (start == i)
                    return error::syntax(tarr[i], "Expected signature after 'fn'");
                if (tarr[start]->ty == TokenType::STAR)
                {
                    ast_arg_decl.is_packed = true;
                    for (start++; start < i && tarr[start]->is_whitespace(); start++);
                    if (start == i)
                        return error::syntax(tarr[i], "Expected signature after 'fn'");
                }
                if (tarr[start]->expect<TokenType::IDN>())
                    return true;
                ast_arg_decl.var_name = tarr[start]->get<TokenType::IDN>().val;
                ast_arg_decl.type_expr = std::make_unique<AstExpr>();
                new (&ast_arg_decl.type_expr->expr_fn_decl) AstFnSig();
                ast_arg_decl.type_expr->ty = ExprType::FN_DECL;
                return parse_signature(tarr, ast_arg_decl.type_expr->expr_fn_decl);
            }

            for (i--; tarr[i]->is_whitespace(); i--);
            if (tarr[i]->expect<TokenType::IDN>())
                return true;
            ast_arg_decl.var_name = tarr[i]->get<TokenType::IDN>().val;
            for (i--; i >= 0 && tarr[i]->is_whitespace(); i--);
            if (i == -1)
            {
                // deduce type from default_expr in the compiler
                ast_arg_decl.type_expr = nullptr;
                return false;
            }

            // checking for a packed argument
            if (tarr[i]->ty == TokenType::STAR)
            {
                ast_arg_decl.is_packed = true;
                for (i--; i >= 0 && tarr[i]->is_whitespace(); i--);
                if (i == -1)
                {
                    // deduce type from default_expr in the compiler
                    ast_arg_decl.type_expr = nullptr;
                    return false;
                }
            }

            ast_arg_decl.type_expr = std::make_unique<AstExpr>();
            return parse_expr({ tarr, 0, i + 1 }, *ast_arg_decl.type_expr);
        }

        template<CodeBlockSig SIG>
        bool parse_code_block(const TokenArray& tarr, AstCodeBlock<SIG>& ast_code_block, int indent_level)
        {
            assert(tarr.size() > 0);
            ast_code_block.node_info = {
                .fname = tarr[0]->fname,
                .line_start = tarr[0]->line_num,
                .line_end = tarr[tarr.size() - 1]->line_num,
                .col_start = tarr[0]->col_num,
                .col_end = tarr[tarr.size() - 1]->col_num
            };

            // Splitting the code block between the signature and the body
            int i = tarr.search(IsSameCriteria(TokenType::COLON));
            if (i == -1)
                return error::syntax(tarr[0], "Missing ':' in signature");
            if (parse_signature({ tarr, 0, i }, ast_code_block.signature))
                return true;

            // Finding the line that starts the body
            assert(!tarr[i]->expect<TokenType::COLON>());
            for (i++; i < tarr.size() && tarr[i]->ty == TokenType::INDENT; i++);  // one token past the last tab after the ':'
            if (i == tarr.size())
                return error::syntax(tarr[0], "Expected new line after ':'");
            if (tarr[i]->expect<TokenType::ENDL>())
                return true;

            // checking if the body is bytecode
            for (i++; tarr[i]->is_whitespace(); i++);
            if (tarr[i]->ty == TokenType::COLON)
            {
                ast_code_block.is_bytecode = true;
                ast_code_block.tarr = std::make_unique<TokenArray>(tarr, i);
                return false;
            }

            // not bytecode, just a normal body that will be compiled
            ast_code_block.is_bytecode = false;
            return parse_lines<AstLine, parse_line>(tarr, i, indent_level, ast_code_block.body);
        }

        bool parse_enum_entry(const TokenArray& tarr, AstEnumEntry& ast_entry, int indent_level)
        {
            assert(tarr.size() > 0);
            ast_entry.node_info = {
                .fname = tarr[0]->fname,
                .line_start = tarr[0]->line_num,
                .line_end = tarr[tarr.size() - 1]->line_num,
                .col_start = tarr[0]->col_num,
                .col_end = tarr[tarr.size() - 1]->col_num
            };

            int i = 0;
            for (; i < tarr.size() && tarr[i]->is_whitespace(); i++);
            if (i == tarr.size())
                return error::syntax(tarr[0], "Empty enum entries are not allowed");

            if (tarr[i]->expect<TokenType::IDN>())
                return true;
            ast_entry.name = tarr[i]->get<TokenType::IDN>().val;
            for (i++; i < tarr.size() && tarr[i]->ty == TokenType::INDENT; i++);
            if (i == tarr.size() || tarr[i]->ty == TokenType::ENDL)
            {
                // Potentially an empty enum entry
                for (; i < tarr.size() && tarr[i]->is_whitespace(); i++);
                if (i == tarr.size())
                    return false;
            }
            if (tarr[i]->expect<TokenType::COLON>())
                return true;
            for (i++; i < tarr.size() && tarr[i]->ty == TokenType::INDENT; i++);  // one token past the last tab after the ':'
            if (i == tarr.size())
                return error::syntax(tarr[0], "Expected new line after ':'");
            if (tarr[i]->expect<TokenType::ENDL>())
                return true;
            return parse_lines<AstLine, parse_line>(tarr, i, indent_level + 1, ast_entry.lines);
        }

        bool parse_enum(const TokenArray& tarr, AstEnum& ast_enum, int indent_level)
        {
            assert(tarr.size() > 0);
            ast_enum.node_info = {
                .fname = tarr[0]->fname,
                .line_start = tarr[0]->line_num,
                .line_end = tarr[tarr.size() - 1]->line_num,
                .col_start = tarr[0]->col_num,
                .col_end = tarr[tarr.size() - 1]->col_num
            };

            // Splitting the enum between the signature and the body
            int i = tarr.search(IsSameCriteria(TokenType::COLON));
            if (i == -1)
                return error::syntax(tarr[0], "Missing ':' in signature");
            if (parse_signature({ tarr, 0, i }, ast_enum.signature))
                return true;

            // Finding the line that starts the body
            assert(!tarr[i]->expect<TokenType::COLON>());
            for (i++; tarr[i]->ty == TokenType::INDENT; i++);  // one token past the last tab after the ':'
            if (tarr[i]->expect<TokenType::ENDL>())
                return true;

            // Parsing out the body of the enum
            if (parse_lines<AstEnumEntry, parse_enum_entry>(tarr, i, indent_level, ast_enum.entries))
                return true;

            // Making sure that no labels are duplicated in the enum, and building out the entry map
            bool ret = false;
            for (const auto& entry : ast_enum.entries)
            {
                const auto& it = ast_enum.entry_map.find(entry.name);
                if (it != ast_enum.entry_map.end())
                    ret = error::syntax(it->second->node_info, "Found duplicate label % in enum definition", it->first);
                else
                    ast_enum.entry_map[entry.name] = &entry;
            }
            return ret;
        }

        bool parse_init(const TokenArray& tarr, AstInit& ast_init)
        {
            assert(tarr.size() > 0);
            ast_init.node_info = {
                .fname = tarr[0]->fname,
                .line_start = tarr[0]->line_num,
                .line_end = tarr[tarr.size() - 1]->line_num,
                .col_start = tarr[0]->col_num,
                .col_end = tarr[tarr.size() - 1]->col_num
            };

            return parse_signature(tarr, ast_init.signature);
        }

        bool parse_import(const TokenArray& tarr, AstImport& ast_import)
        {
            assert(tarr.size() > 0);
            ast_import.node_info = {
                .fname = tarr[0]->fname,
                .line_start = tarr[0]->line_num,
                .line_end = tarr[tarr.size() - 1]->line_num,
                .col_start = tarr[0]->col_num,
                .col_end = tarr[tarr.size() - 1]->col_num
            };

            bool expect_idn = true;
            for (auto& tk : tarr)
            {
                if (tk.is_whitespace())
                    continue;
                if (expect_idn)
                {
                    if (tk.expect<TokenType::IDN>())
                        return true;
                    ast_import.imp.push_back(tk.get<TokenType::IDN>().val);
                    expect_idn = false;
                }
                else
                {
                    if (tk.expect<TokenType::DOT>())
                        return true;
                    expect_idn = true;
                }
            }
            return false;
        }

        bool parse_namespace(const TokenArray& tarr, AstNamespace& ast_namespace, int indent_level)
        {
            int i = 0;
            for (; i < tarr.size() && tarr[i]->is_whitespace(); i++);
            if (tarr[i]->expect<TokenType::IDN>())
                return true;
            ast_namespace.name = tarr[i]->get<TokenType::IDN>().val;
            for (i++; i < tarr.size() && tarr[i]->is_whitespace(); i++);
            if (tarr[i]->expect<TokenType::COLON>())
                return true;
            i++;

            bool ret = false;
            for (; i < tarr.size(); i++)
            {
                if (tarr[i]->ty == TokenType::ENDL || tarr[i]->ty == TokenType::INDENT)
                    continue;

                int end;
                switch (tarr[i]->ty)
                {
                case TokenType::KW_IMPORT:
                    ret = error::syntax(tarr[i], "imports are not allowed within a namespace");
                    i = tarr.search(IsSameCriteria(TokenType::ENDL), i);
                    break;
                case TokenType::KW_NAMESPACE:
                    i++;
                    end = tarr.search(LineEndCriteria(indent_level), i);
                    if (end == i)
                        return error::syntax(tarr[i], "Expected identifier after 'namespace'");
                    if (end < 0)
                        end = tarr.size();
                    ast_namespace.namespaces.push_back(AstNamespace());
                    if (ret = parse_namespace({ tarr, i, end }, ast_namespace.namespaces.back(), indent_level + 1) || ret)
                        ast_namespace.namespaces.pop_back();
                    break;
                case TokenType::KW_STRUCT:
                    i++;
                    end = tarr.search(LineEndCriteria(indent_level), i);
                    if (end == i)
                        return error::syntax(tarr[i], "Expected identifier after 'struct'");
                    if (end < 0)
                        end = tarr.size();
                    ast_namespace.structs.push_back(AstStruct());
                    if (ret = parse_code_block({ tarr, i, end }, ast_namespace.structs.back(), indent_level + 1) || ret)
                        ast_namespace.structs.pop_back();
                    break;
                case TokenType::KW_ENUM:
                    i++;
                    end = tarr.search(LineEndCriteria(indent_level), i);
                    if (end == i)
                        return error::syntax(tarr[i], "Expected identifier after 'enum'");
                    if (end < 0)
                        end = tarr.size();
                    ast_namespace.enums.push_back(AstEnum());
                    if (ret = parse_enum({ tarr, i, end }, ast_namespace.enums.back(), indent_level + 1) || ret)
                        ast_namespace.enums.pop_back();
                    break;
                case TokenType::KW_FN:
                    i++;
                    end = tarr.search(LineEndCriteria(indent_level), i);
                    if (end == i)
                        return error::syntax(tarr[i], "Expected identifier after 'fn'");
                    if (end < 0)
                        end = tarr.size();
                    ast_namespace.funcs.push_back(AstFn());
                    if (ret = parse_code_block({ tarr, i, end }, ast_namespace.funcs.back(), indent_level + 1) || ret)
                        ast_namespace.funcs.pop_back();
                    break;
                case TokenType::KW_DEF:
                    i++;
                    end = tarr.search(LineEndCriteria(indent_level), i);
                    if (end == i)
                        return error::syntax(tarr[i], "Expected identifier after 'def'");
                    if (end < 0)
                        end = tarr.size();
                    ast_namespace.defs.push_back(AstBlock());
                    if (ret = parse_code_block({ tarr, i, end }, ast_namespace.defs.back(), indent_level + 1) || ret)
                        ast_namespace.defs.pop_back();
                    break;
                case TokenType::KW_INTR:
                    i++;
                    end = tarr.search(LineEndCriteria(indent_level), i);
                    if (end == i)
                        return error::syntax(tarr[i], "Expected identifier after 'intr'");
                    if (end < 0)
                        end = tarr.size();
                    ast_namespace.intrs.push_back(AstBlock());
                    if (ret = parse_code_block({ tarr, i, end }, ast_namespace.intrs.back(), indent_level + 1) || ret)
                        ast_namespace.intrs.pop_back();
                    break;
                case TokenType::KW_INIT:
                    i++;
                    end = tarr.search(IsSameCriteria(TokenType::ENDL), i);
                    if (end == i)
                        return error::syntax(tarr[i], "Expected identifier after 'init'");
                    if (end < 0)
                        end = tarr.size();
                    ast_namespace.inits.push_back(AstInit());
                    if (ret = parse_init({ tarr, i, end }, ast_namespace.inits.back()) || ret)
                        ast_namespace.inits.pop_back();
                    break;
                default:
                    return error::syntax(tarr[i], "Invalid token in namespace");
                }
                i = end;
            }
            return ret;
        }

        bool parse_module(const TokenArray& tarr, AstModule& ast_module)
        {
            ast_module.fname = tarr[0]->fname;

            bool ret = false;
            for (int i = 0; i < tarr.size(); i++)
            {
                if (tarr[i]->ty == TokenType::ENDL || tarr[i]->ty == TokenType::INDENT)
                    continue;
                
                int end;
                switch (tarr[i]->ty)
                {
                case TokenType::KW_IMPORT:
                    i++;
                    end = tarr.search(IsSameCriteria(TokenType::ENDL), i);
                    if (end == i)
                        return error::syntax(tarr[i], "Expected identifier after 'import'");
                    if (end < 0)
                        end = tarr.size();
                    ast_module.imports.push_back(AstImport());
                    if (ret = parse_import({ tarr, i, end }, ast_module.imports.back()) || ret)
                        ast_module.imports.pop_back();
                    break;
                case TokenType::KW_NAMESPACE:
                    i++;
                    end = tarr.search(LineEndCriteria(0), i);
                    if (end == i)
                        return error::syntax(tarr[i], "Expected identifier after 'namespace'");
                    if (end < 0)
                        end = tarr.size();
                    ast_module.namespaces.push_back(AstNamespace());
                    if (ret = parse_namespace({ tarr, i, end }, ast_module.namespaces.back(), 1) || ret)
                        ast_module.namespaces.pop_back();
                    break;
                case TokenType::KW_STRUCT:
                    i++;
                    end = tarr.search(LineEndCriteria(0), i);
                    if (end == i)
                        return error::syntax(tarr[i], "Expected identifier after 'struct'");
                    if (end < 0)
                        end = tarr.size();
                    ast_module.structs.push_back(AstStruct());
                    if (ret = parse_code_block({ tarr, i, end }, ast_module.structs.back(), 1) || ret)
                        ast_module.structs.pop_back();
                    break;
                case TokenType::KW_ENUM:
                    i++;
                    end = tarr.search(LineEndCriteria(0), i);
                    if (end == i)
                        return error::syntax(tarr[i], "Expected identifier after 'enum'");
                    if (end < 0)
                        end = tarr.size();
                    ast_module.enums.push_back(AstEnum());
                    if (ret = parse_enum({ tarr, i, end }, ast_module.enums.back(), 1) || ret)
                        ast_module.enums.pop_back();
                    break;
                case TokenType::KW_FN:
                    i++;
                    end = tarr.search(LineEndCriteria(0), i);
                    if (end == i)
                        return error::syntax(tarr[i], "Expected identifier after 'fn'");
                    if (end < 0)
                        end = tarr.size();
                    ast_module.funcs.push_back(AstFn());
                    if (ret = parse_code_block({ tarr, i, end }, ast_module.funcs.back(), 1) || ret)
                        ast_module.funcs.pop_back();
                    break;
                case TokenType::KW_DEF:
                    i++;
                    end = tarr.search(LineEndCriteria(0), i);
                    if (end == i)
                        return error::syntax(tarr[i], "Expected identifier after 'def'");
                    if (end < 0)
                        end = tarr.size();
                    ast_module.defs.push_back(AstBlock());
                    if (ret = parse_code_block({ tarr, i, end }, ast_module.defs.back(), 1) || ret)
                        ast_module.defs.pop_back();
                    break;
                case TokenType::KW_INTR:
                    i++;
                    end = tarr.search(LineEndCriteria(0), i);
                    if (end == i)
                        return error::syntax(tarr[i], "Expected identifier after 'intr'");
                    if (end < 0)
                        end = tarr.size();
                    ast_module.intrs.push_back(AstBlock());
                    if (ret = parse_code_block({ tarr, i, end }, ast_module.intrs.back(), 1) || ret)
                        ast_module.intrs.pop_back();
                    break;
                case TokenType::KW_INIT:
                    i++;
                    end = tarr.search(IsSameCriteria(TokenType::ENDL), i);
                    if (end == i)
                        return error::syntax(tarr[i], "Expected identifier after 'init'");
                    if (end < 0)
                        end = tarr.size();
                    ast_module.inits.push_back(AstInit());
                    if (ret = parse_init({ tarr, i, end }, ast_module.inits.back()) || ret)
                        ast_module.inits.pop_back();
                    break;
                default:
                    return error::syntax(tarr[i], "Invalid token in module");
                }
                i = end;
            }
            return ret;
        }

        // From this point onward, the actual code ends and we enter c++ land

        AstExpr::AstExpr() {}

        AstExpr::AstExpr(AstExpr&& expr)
        {
            do_move(std::move(expr));
        }

        AstExpr& AstExpr::operator=(AstExpr&& expr) noexcept
        {
            if (&expr == this)
                return *this;
            this->~AstExpr();
            do_move(std::move(expr));
            return *this;
        }

        AstExpr::~AstExpr()
        {
            switch (ty)
            {
            case ExprType::INVALID:
            case ExprType::VOID:
            case ExprType::LIT_BOOL:
            case ExprType::LIT_INT:
            case ExprType::LIT_FLOAT:
            case ExprType::KW:
                break;
            case ExprType::LIT_STRING:
            case ExprType::VAR:
                expr_string.~basic_string();
                break;
            case ExprType::LIT_ARRAY:
            case ExprType::LIT_TUPLE:
                expr_agg.~AstExprAggLit();
                break;
            case ExprType::UNARY_POS:
            case ExprType::UNARY_NEG:
            case ExprType::UNARY_NOT:
            case ExprType::UNARY_UNPACK:
            case ExprType::UNARY_REF:
            case ExprType::UNARY_CONST:
            case ExprType::UNARY_FORWARD:
            case ExprType::UNARY_BACKWARD:
                expr_unary.~AstExprUnaryOp();
                break;
            case ExprType::BINARY_ADD:
            case ExprType::BINARY_SUB:
            case ExprType::BINARY_MUL:
            case ExprType::BINARY_DIV:
            case ExprType::BINARY_MOD:
            case ExprType::BINARY_POW:
            case ExprType::BINARY_IADD:
            case ExprType::BINARY_ISUB:
            case ExprType::BINARY_IMUL:
            case ExprType::BINARY_IDIV:
            case ExprType::BINARY_IMOD:
            case ExprType::BINARY_IPOW:
            case ExprType::BINARY_ASSIGN:
            case ExprType::BINARY_AND:
            case ExprType::BINARY_OR:
            case ExprType::BINARY_CMP_EQ:
            case ExprType::BINARY_CMP_NE:
            case ExprType::BINARY_CMP_GT:
            case ExprType::BINARY_CMP_LT:
            case ExprType::BINARY_CMP_GE:
            case ExprType::BINARY_CMP_LE:
            case ExprType::BINARY_CAST:
                expr_binary.~AstExprBinaryOp();
                break;
            case ExprType::INDEX:
                expr_index.~AstExprIndex();
                break;
            case ExprType::DOT:
            case ExprType::VAR_DECL:
                expr_name.~AstExprName();
                break;
            case ExprType::CARGS_CALL:
            case ExprType::VARGS_CALL:
                expr_call.~AstExprCall();
                break;
            case ExprType::DEF_DECL:
            case ExprType::INTR_DECL:
                expr_blk_decl.~AstBlockSig();
                break;
            case ExprType::FN_DECL:
                expr_fn_decl.~AstFnSig();
                break;
            default:
                assert(false);
            }
        }

        void AstExpr::do_move(AstExpr&& expr) noexcept
        {
            ty = expr.ty;
            node_info = std::move(expr.node_info);
            expr.ty = ExprType::INVALID;

            switch (ty)
            {
            case ExprType::INVALID:
            case ExprType::VOID:
                break;
            case ExprType::LIT_BOOL:
                new (&expr_bool) decltype(expr_bool)(std::move(expr.expr_bool));
                break;
            case ExprType::LIT_INT:
                new (&expr_int) decltype(expr_int)(std::move(expr.expr_int));
                break;
            case ExprType::LIT_FLOAT:
                new (&expr_float) decltype(expr_float)(std::move(expr.expr_float));
                break;
            case ExprType::KW:
                new (&expr_kw) decltype(expr_kw)(std::move(expr.expr_kw));
                break;
            case ExprType::LIT_STRING:
            case ExprType::VAR:
                new (&expr_string) decltype(expr_string)(std::move(expr.expr_string));
                break;
            case ExprType::LIT_ARRAY:
            case ExprType::LIT_TUPLE:
                new (&expr_agg) decltype(expr_agg)(std::move(expr.expr_agg));
                break;
            case ExprType::UNARY_POS:
            case ExprType::UNARY_NEG:
            case ExprType::UNARY_NOT:
            case ExprType::UNARY_UNPACK:
            case ExprType::UNARY_REF:
            case ExprType::UNARY_CONST:
            case ExprType::UNARY_FORWARD:
            case ExprType::UNARY_BACKWARD:
                new (&expr_unary) decltype(expr_unary)(std::move(expr.expr_unary));
                break;
            case ExprType::BINARY_ADD:
            case ExprType::BINARY_SUB:
            case ExprType::BINARY_MUL:
            case ExprType::BINARY_DIV:
            case ExprType::BINARY_MOD:
            case ExprType::BINARY_POW:
            case ExprType::BINARY_IADD:
            case ExprType::BINARY_ISUB:
            case ExprType::BINARY_IMUL:
            case ExprType::BINARY_IDIV:
            case ExprType::BINARY_IMOD:
            case ExprType::BINARY_IPOW:
            case ExprType::BINARY_ASSIGN:
            case ExprType::BINARY_AND:
            case ExprType::BINARY_OR:
            case ExprType::BINARY_CMP_EQ:
            case ExprType::BINARY_CMP_NE:
            case ExprType::BINARY_CMP_GT:
            case ExprType::BINARY_CMP_LT:
            case ExprType::BINARY_CMP_GE:
            case ExprType::BINARY_CMP_LE:
            case ExprType::BINARY_CAST:
                new (&expr_binary) decltype(expr_binary)(std::move(expr.expr_binary));
                break;
            case ExprType::INDEX:
                new (&expr_index) decltype(expr_index)(std::move(expr.expr_index));
                break;
            case ExprType::DOT:
            case ExprType::VAR_DECL:
                new (&expr_name) decltype(expr_name)(std::move(expr.expr_name));
                break;
            case ExprType::CARGS_CALL:
            case ExprType::VARGS_CALL:
                new (&expr_call) decltype(expr_call)(std::move(expr.expr_call));
                break;
            case ExprType::DEF_DECL:
            case ExprType::INTR_DECL:
                new (&expr_blk_decl) decltype(expr_blk_decl)(std::move(expr.expr_blk_decl));
            case ExprType::FN_DECL:
                new (&expr_fn_decl) decltype(expr_fn_decl)(std::move(expr.expr_fn_decl));
                break;
                break;
            default:
                assert(false);
            }
        }

        AstLine::AstLine() {}

        AstLine::AstLine(AstLine&& line) noexcept
        {
            do_move(std::move(line));
        }

        AstLine& AstLine::operator=(AstLine&& line) noexcept
        {
            if (&line == this)
                return *this;
            this->~AstLine();
            do_move(std::move(line));
            return *this;
        }

        AstLine::~AstLine()
        {
            switch (ty)
            {
            case LineType::INVALID:
            case LineType::BREAK:
            case LineType::CONTINUE:
                break;
            case LineType::EXPORT:
                line_export.~AstLineExport();
                break;
            case LineType::EXTERN:
                line_extern.~AstLineExtern();
                break;
            case LineType::RETURN:
                line_func.~AstLineUnaryFunc();
                break;
            case LineType::INTRINFO:
                line_intrinfo.~AstLineIntrInfo();
                break;
            case LineType::MATCH:
                line_match.~AstLineMatch();
                break;
            case LineType::IF:
            case LineType::ELIF:
            case LineType::WHILE:
                line_branch.~AstLineBranch();
                break;
            case LineType::ELSE:
                line_block.~AstLineBlock();
                break;
            case LineType::FOR:
                line_for.~AstLineFor();
                break;
            case LineType::EXPR:
                line_expr.~AstLineExpr();
                break;
            case LineType::EVALMODE:
                line_label.~AstLineLabel();
                break;
            default:
                assert(false);
            }
        }

        void AstLine::do_move(AstLine&& line) noexcept
        {
            ty = line.ty;
            node_info = std::move(line.node_info);
            line.ty = LineType::INVALID;

            switch (ty)
            {
            case LineType::INVALID:
            case LineType::BREAK:
            case LineType::CONTINUE:
                break;
            case LineType::EXPORT:
                new (&line_export) decltype(line_export)(std::move(line.line_export));
                break;
            case LineType::EXTERN:
                new (&line_extern) decltype(line_extern)(std::move(line.line_extern));
                break;
            case LineType::RETURN:
                new (&line_func) decltype(line_func)(std::move(line.line_func));
                break;
            case LineType::INTRINFO:
                new (&line_intrinfo) decltype(line_intrinfo)(std::move(line.line_intrinfo));
                break;
            case LineType::MATCH:
                new (&line_match) decltype(line_match)(std::move(line.line_match));
                break;
            case LineType::IF:
            case LineType::ELIF:
            case LineType::WHILE:
                new (&line_branch) decltype(line_branch)(std::move(line.line_branch));
                break;
            case LineType::ELSE:
                new (&line_block) decltype(line_block)(std::move(line.line_block));
                break;
            case LineType::FOR:
                new (&line_for) decltype(line_for)(std::move(line.line_for));
                break;
            case LineType::EXPR:
                new (&line_expr) decltype(line_expr)(std::move(line.line_expr));
                break;
            case LineType::EVALMODE:
                new (&line_label) decltype(line_label)(std::move(line.line_label));
                break;
            default:
                assert(false);
            }
        }
    }
}
