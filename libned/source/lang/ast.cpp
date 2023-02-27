#include <ned/errors.h>
#include <ned/lang/ast.h>
#include <ned/lang/lexer.h>
#include <ned/lang/obj.h>
#include <ned/lang/bytecode.h>

#include <functional>
#include <cassert>

#include <array>
#include <iostream>

namespace nn
{
    namespace lang
    {
        constexpr auto token_expr_table = []
        {
            std::array<AstExpr::Type, (size_t)TokenType::TOKEN_TYPE_END> table = {};

            // Mapping tokens to expression types
            table[(size_t)TokenType::COLON  ] = AstExpr::Type::BINARY_DECL;
            table[(size_t)TokenType::CAST   ] = AstExpr::Type::BINARY_CAST;
            table[(size_t)TokenType::ADD    ] = AstExpr::Type::BINARY_ADD;
            table[(size_t)TokenType::SUB    ] = AstExpr::Type::BINARY_SUB;
            table[(size_t)TokenType::STAR   ] = AstExpr::Type::BINARY_MUL;
            table[(size_t)TokenType::DIV    ] = AstExpr::Type::BINARY_DIV;
            table[(size_t)TokenType::MOD    ] = AstExpr::Type::BINARY_MOD;
            table[(size_t)TokenType::POW    ] = AstExpr::Type::BINARY_POW;
            table[(size_t)TokenType::IADD   ] = AstExpr::Type::BINARY_IADD;
            table[(size_t)TokenType::ISUB   ] = AstExpr::Type::BINARY_ISUB;
            table[(size_t)TokenType::IMUL   ] = AstExpr::Type::BINARY_IMUL;
            table[(size_t)TokenType::IDIV   ] = AstExpr::Type::BINARY_IDIV;
            table[(size_t)TokenType::IMOD   ] = AstExpr::Type::BINARY_IMOD;
            table[(size_t)TokenType::IPOW   ] = AstExpr::Type::BINARY_IPOW;
            table[(size_t)TokenType::ASSIGN ] = AstExpr::Type::BINARY_ASSIGN;
            table[(size_t)TokenType::CMP_EQ ] = AstExpr::Type::BINARY_CMP_EQ;
            table[(size_t)TokenType::CMP_NE ] = AstExpr::Type::BINARY_CMP_NE;
            table[(size_t)TokenType::CMP_GT ] = AstExpr::Type::BINARY_CMP_GT;
            table[(size_t)TokenType::CMP_LT ] = AstExpr::Type::BINARY_CMP_LT;
            table[(size_t)TokenType::CMP_GE ] = AstExpr::Type::BINARY_CMP_GE;
            table[(size_t)TokenType::CMP_LE ] = AstExpr::Type::BINARY_CMP_LE;
            table[(size_t)TokenType::KW_AND ] = AstExpr::Type::BINARY_AND;
            table[(size_t)TokenType::KW_OR  ] = AstExpr::Type::BINARY_OR;

            return table;
        }();

        template<PrecTable prec_table>
        class PrioritySplitCriteria :
            public BracketCounter
        {
        public:
            int accept(const Token* ptk, int idx)
            {
                count_token(ptk);
                if (in_bracket() || prec_table[(size_t)ptk->ty] == 0)
                    return -1;
                return idx;
            }
        };

        std::string to_string(ExprKW kw)
        {
            switch (kw)
            {
            case ExprKW::NUL:
                return "null";
            case ExprKW::TYPE:
                return "type";
            case ExprKW::VOID:
                return "void";
            case ExprKW::INIT:
                return "init";
            case ExprKW::FTY:
                return "fty";
            case ExprKW::BOOL:
                return "bool";
            case ExprKW::INT:
                return "int";
            case ExprKW::FLOAT:
                return "float";
            case ExprKW::STR:
                return "str";
            case ExprKW::CFG:
                return "cfg";
            case ExprKW::ARRAY:
                return "array";
            case ExprKW::TUPLE:
                return "tuple";
            case ExprKW::F16:
                return "f16";
            case ExprKW::F32:
                return "f32";
            case ExprKW::F64:
                return "f64";
            default:
                return "INVALID";
            }
        }

        // Parses lines from a token sequence
        // On failure, this function returns true
        template<typename T, bool(*parse_fn)(const TokenArray&, T&, int)>
        inline bool parse_lines(const TokenArray& tarr, int indent_level, std::vector<T>& lines)
        {
            int i = 0;
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

        // Parses comma deliminated token sequences within a matching set of brackets
        // Returns the index in tarr immediately after the CLOSE token was found
        inline bool parse_args(const TokenArray& tarr, std::vector<AstExpr>& args, int indent_level)
        {
            // Taking advantage of the existing tuple parsing code in parse_expr
            AstExpr args_expr;
            if (parse_expr<args_prec_table, args_rtol_table>(tarr, args_expr, indent_level))
                return error::syntax(tarr[0], "Error parsing cargs");
            // Dealing with the result of parse_expr
            if (args_expr.ty == AstExpr::Type::EMPTY)
                return false;  // No cargs
            if (args_expr.ty == AstExpr::Type::LIT_TUPLE)
            {
                // Multiple cargs
                args = std::move(args_expr.expr_agg.elems);
                return false;
            }
            // Single carg
            args.push_back(std::move(args_expr));
            return false;
        }

        bool parse_index(const TokenArray& tarr, std::vector<AstExprIndex::Elem>& elems, int ilv)
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
                    if (parse_expr({ tarr, start, end }, *elem.lhs, ilv + 1))
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
                        if (parse_expr({ tarr, start, mid }, *elem.lhs, ilv + 1))
                            return true;
                    }
                    for (mid++; mid < end && tarr[mid]->is_whitespace(); mid++);
                    if (mid != end)
                    {
                        // Similar logic to the if (start != mid) thing.  Even in the case where
                        // both the lhs and rhs are null, its still valid syntax
                        elem.rhs = std::make_unique<AstExpr>();
                        if (parse_expr({ tarr, mid, end }, *elem.rhs, ilv + 1))
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

        bool parse_leaf_mods(const TokenArray& tarr, AstExpr& ast_expr, std::unique_ptr<AstExpr> lhs, int ilv)
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
                    pexpr->ty = AstExpr::Type::CALL_CARGS;
                    pexpr->expr_call.callee = std::move(lhs);
                    break;
                case TokenType::ROUND_O:
                    new (&pexpr->expr_call) AstExprCall();
                    pexpr->ty = AstExpr::Type::CALL_VARGS;
                    pexpr->expr_call.callee = std::move(lhs);
                    break;
                default:
                    return error::syntax(tarr[start], "Empty expression");
                }
                if (i == tarr.size())
                    return false;
                return parse_leaf_mods({ tarr, i }, ast_expr, std::move(nlhs), ilv);
            }

            AstExpr ret_expr;  // helper for calls so that I leverge the tuple parsing code for this
            switch (tarr[0]->ty)
            {
            case TokenType::SQUARE_O:
                // TODO: implement index with slicing
                new (&pexpr->expr_binary) AstExprIndex();
                pexpr->ty = AstExpr::Type::INDEX;
                pexpr->expr_index.expr = std::move(lhs);
                if (parse_index({ tarr, start, end }, pexpr->expr_index.args, ilv))
                    return true;
                break;
            case TokenType::DOT:
                if (tarr[start]->expect<TokenType::IDN>())
                    return true;
                new (&pexpr->expr_name) AstExprName();
                pexpr->ty = AstExpr::Type::DOT;
                pexpr->expr_name.expr = std::move(lhs);
                pexpr->expr_name.val = tarr[start]->get<TokenType::IDN>().val;
                break;
            case TokenType::ANGLE_O:
                new (&pexpr->expr_call) AstExprCall();
                pexpr->ty = AstExpr::Type::CALL_CARGS;
                pexpr->expr_call.callee = std::move(lhs);
                if (parse_expr({ tarr, start, end }, ret_expr, ilv + 1))
                    return true;
                if (ret_expr.ty == AstExpr::Type::LIT_TUPLE)
                    pexpr->expr_call.args = std::move(ret_expr.expr_agg.elems);
                else
                    pexpr->expr_call.args.push_back(std::move(ret_expr));
                break;
            case TokenType::ROUND_O:
                new (&pexpr->expr_call) AstExprCall();
                pexpr->ty = AstExpr::Type::CALL_VARGS;
                pexpr->expr_call.callee = std::move(lhs);
                if (parse_expr({ tarr, start, end }, ret_expr, ilv + 1))
                    return true;
                if (ret_expr.ty == AstExpr::Type::LIT_TUPLE)
                    pexpr->expr_call.args = std::move(ret_expr.expr_agg.elems);
                else
                    pexpr->expr_call.args.push_back(std::move(ret_expr));
                break;
            default:
                assert(false);
            }

            if (i == tarr.size())
                return false;
            return parse_leaf_mods({ tarr, i }, ast_expr, std::move(nlhs), ilv);
        }

        bool parse_leaf_token(const Token* ptk, AstExpr& ast_expr)
        {
            switch (ptk->ty)
            {
            case TokenType::LIT_INT:
                ast_expr.expr_int = ptk->get<TokenType::LIT_INT>().val;
                ast_expr.ty = AstExpr::Type::LIT_INT;
                return false;
            case TokenType::LIT_FLOAT:
                ast_expr.expr_float = ptk->get<TokenType::LIT_FLOAT>().val;
                ast_expr.ty = AstExpr::Type::LIT_FLOAT;
                return false;
            case TokenType::LIT_STR:
                new (&ast_expr.expr_string) std::string(ptk->get<TokenType::LIT_STR>().val);
                ast_expr.ty = AstExpr::Type::LIT_STRING;
                return false;
            case TokenType::IDN:
                new (&ast_expr.expr_string) std::string(ptk->get<TokenType::IDN>().val);
                ast_expr.ty = AstExpr::Type::VAR;
                return false;
            case TokenType::KW_TRUE:
                ast_expr.expr_bool = true;
                ast_expr.ty = AstExpr::Type::LIT_BOOL;
                return false;
            case TokenType::KW_FALSE:
                ast_expr.expr_bool = false;
                ast_expr.ty = AstExpr::Type::LIT_BOOL;
                return false;
            case TokenType::KW_NULL:
                ast_expr.expr_kw = ExprKW::NUL;
                ast_expr.ty = AstExpr::Type::KW;
                return false;
            case TokenType::KW_TYPE:
                ast_expr.expr_kw = ExprKW::TYPE;
                ast_expr.ty = AstExpr::Type::KW;
                return false;
            case TokenType::KW_VOID:
                ast_expr.expr_kw = ExprKW::VOID;
                ast_expr.ty = AstExpr::Type::KW;
                return false;
            case TokenType::KW_INIT:
                ast_expr.expr_kw = ExprKW::INIT;
                ast_expr.ty = AstExpr::Type::KW;
                return false;
            case TokenType::KW_FTY:
                ast_expr.expr_kw = ExprKW::FTY;
                ast_expr.ty = AstExpr::Type::KW;
                return false;
            case TokenType::KW_BOOL:
                ast_expr.expr_kw = ExprKW::BOOL;
                ast_expr.ty = AstExpr::Type::KW;
                return false;
            case TokenType::KW_INT:
                ast_expr.expr_kw = ExprKW::INT;
                ast_expr.ty = AstExpr::Type::KW;
                return false;
            case TokenType::KW_FLOAT:
                ast_expr.expr_kw = ExprKW::FLOAT;
                ast_expr.ty = AstExpr::Type::KW;
                return false;
            case TokenType::KW_STR:
                ast_expr.expr_kw = ExprKW::STR;
                ast_expr.ty = AstExpr::Type::KW;
                return false;
            case TokenType::KW_ARRAY:
                ast_expr.expr_kw = ExprKW::ARRAY;
                ast_expr.ty = AstExpr::Type::KW;
                return false;
            case TokenType::KW_TUPLE:
                ast_expr.expr_kw = ExprKW::TUPLE;
                ast_expr.ty = AstExpr::Type::KW;
                return false;
            case TokenType::KW_CFG:
                ast_expr.expr_kw = ExprKW::CFG;
                ast_expr.ty = AstExpr::Type::KW;
                return false;
            case TokenType::KW_F16:
                ast_expr.expr_kw = ExprKW::F16;
                ast_expr.ty = AstExpr::Type::KW;
                return false;
            case TokenType::KW_F32:
                ast_expr.expr_kw = ExprKW::F32;
                ast_expr.ty = AstExpr::Type::KW;
                return false;
            case TokenType::KW_F64:
                ast_expr.expr_kw = ExprKW::F64;
                ast_expr.ty = AstExpr::Type::KW;
                return false;
            default:
                error::syntax(ptk, "Unexpected token for single token expression leaf node");
                return true;
            }
        }

        bool parse_leaf_expr(const TokenArray& tarr, AstExpr& ast_expr, int ilv)
        {
            assert(tarr.size() > 0);

            int i = 0;
            for (; i < tarr.size() && tarr[i]->is_whitespace(); i++);
            if (i == tarr.size())
                return error::syntax(tarr[0], "Empty expression are not allowed");

            if (tarr.size() == 1)  // single token leaf node
                return parse_leaf_token(tarr[0], ast_expr);

            std::unique_ptr<AstExpr> lhs = nullptr;
            if (tarr[i]->ty == TokenType::ROUND_O)
            {
                // handling bracketed expressions
                int end = tarr.search(IsSameCriteria(TokenType::ROUND_C), i);
                if (end == -1)
                    return error::syntax(tarr[i], "Missing closing ')'");
                if (end == i + 1)  // TODO: make it an empty expression
                    return error::syntax(tarr[i], "Empty expression are not allowed");

                // Checking if all the remaining tokens are whitespace
                // This is used to determine if any mods exist on the expression
                bool has_mods = false;
                for (const auto& tk : TokenArray(tarr, end + 1))
                    if (!tk.is_whitespace())
                    {
                        has_mods = true;
                        break;
                    }
                if (!has_mods)
                    return parse_expr({ tarr, i, end }, ast_expr, ilv);

                lhs = std::make_unique<AstExpr>();
                if (parse_expr({ tarr, i, end }, *lhs, ilv))
                    return true;
                i = end;
                for (i++; i < tarr.size() && tarr[i]->is_whitespace(); i++);
            }
            else if (tarr[i]->ty == TokenType::SQUARE_O)
            {
                // handling array literals
                int end = tarr.search(IsSameCriteria(TokenType::SQUARE_C), i);
                if (end == -1)
                    return error::syntax(tarr[i], "Missing closing ']'");

                // Checking if all the remaining tokens are whitespace
                // This is used to determine if any mods exist on the array literal
                bool has_mods = false;
                for (const auto& tk : TokenArray(tarr, end + 1))
                    if (!tk.is_whitespace())
                    {
                        has_mods = true;
                        break;
                    }

                // Checking if the array is empty
                bool is_empty = true;
                for (const auto& tk : TokenArray(tarr, i + 1, end))
                    if (!tk.is_whitespace())
                    {
                        is_empty = false;
                        break;
                    }

                AstExpr* arr_expr = &ast_expr;
                if (has_mods)
                {
                    // has mods => initialize lhs and continue execution
                    lhs = std::make_unique<AstExpr>();
                    lhs->node_info = {
                        .fname = tarr[i]->fname,
                        .line_start = tarr[0]->line_num,
                        .line_end = tarr[end]->line_num,
                        .col_start = tarr[i]->col_num,
                        .col_end = tarr[end]->col_num
                    };
                    arr_expr = lhs.get();
                }
                new (&arr_expr->expr_agg) AstExprAggLit();
                arr_expr->ty = AstExpr::Type::LIT_ARRAY;

                if (is_empty && !has_mods)
                    return false;

                if (!is_empty)
                {
                    if (parse_args({tarr, i + 1, end}, arr_expr->expr_agg.elems, ilv + 1))
                        return true;
                    if (!has_mods)
                        return false;
                }
                for (i = end + 1; i < tarr.size() && tarr[i]->is_whitespace(); i++);
            }
            else
            {
                // single token followed by modifiers
                lhs = std::make_unique<AstExpr>();
                lhs->node_info = {
                    .fname = tarr[i]->fname,
                    .line_start = tarr[i]->line_num,
                    .line_end = tarr[i]->line_num,
                    .col_start = tarr[i]->col_num,
                    .col_end = tarr[i]->col_num
                };
                if (parse_leaf_token(tarr[i], *lhs))
                    return true;
                i++;
            }

            // If execution falls through the above code,
            // it means lhs and start are initialized and there are leaf mods that need to be parsed
            // start should be the index immediately after the lhs expression
            return parse_leaf_mods({ tarr, i }, ast_expr, std::move(lhs), ilv);
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
                parse_lines<AstLine, parse_line>({ tarr, end }, indent_level + 1, ast_match_elem.body);
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
                if (parse_expr({ tarr, i + 1, end }, ast_line.line_extern.init_expr, indent_level))
                    return error::syntax(tarr[1], "Invalid syntax in weight init expression");
                break;

            case TokenType::KW_RETURN:
                new (&ast_line.line_func) AstLineUnaryFunc();
                ast_line.ty = LineType::RETURN;
                if (parse_expr({ tarr, i + 1 }, ast_line.line_func.expr, indent_level))
                    return error::syntax(tarr[i], "Invalid syntax in return statement");
                break;

            case TokenType::KW_ADD_CFG_INFO:
                new (&ast_line.line_cfginfo) AstLineCfgInfo();
                ast_line.ty = LineType::CFGINFO;

                end = tarr.search(IsSameCriteria(TokenType::COLON), i + 1);
                if (end == -1)
                    return error::syntax(tarr[i], "Missing ':' in __add_cfg_info statement");
                if (end == i + 1)
                    return error::syntax(tarr[i], "Empty name expression in __add_cfg_info statement");
                if (parse_expr({ tarr, i + 1, end }, ast_line.line_cfginfo.name_expr, indent_level))
                    return error::syntax(tarr[i], "Invalid syntax in __add_cfg_info name expression");
                if (end + 1 == tarr.size())
                    return error::syntax(tarr[(size_t)end - 1], "Empty cfg expression in __add_cfg_info statement");
                if (parse_expr({ tarr, end + 1 }, ast_line.line_cfginfo.cfg_expr, indent_level))
                    return error::syntax(tarr[end + 1], "Invalid syntax in __add_cfg_info cfg expression");
                return false;

            case TokenType::KW_MATCH:
                new (&ast_line.line_match) AstLineMatch();
                ast_line.ty = LineType::MATCH;

                end = tarr.search(IsSameCriteria(TokenType::COLON), i + 1);
                if (end == -1)
                    return error::syntax(tarr[i], "Missing ':' in match statement");
                if (end == i + 1)
                    return error::syntax(tarr[i], "Empty expression in match condition");
                if (parse_expr({ tarr, i + 1, end }, ast_line.line_match.arg, indent_level))
                    return error::syntax(tarr[i], "Invalid syntax in match argument");

                for (end++; end < tarr.size() && tarr[end]->ty == TokenType::INDENT; end++);
                if (end == tarr.size())
                    return error::syntax(tarr[(size_t)end - 1], "Missing body of match block");
                if (tarr[end]->expect<TokenType::ENDL>() ||
                    parse_lines<AstMatchElem, parse_match_elem>({ tarr, end }, indent_level + 1, ast_line.line_match.elems))
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
                if (parse_expr({ tarr, i + 1, end }, ast_line.line_branch.cond, indent_level))
                    return error::syntax(tarr[i], "Invalid syntax in if condition");

                for (end++; end < tarr.size() && tarr[end]->ty == TokenType::INDENT; end++);
                if (end == tarr.size())
                    return error::syntax(tarr[(size_t)end - 1], "Missing body of evaluation mode block");
                if (tarr[end]->expect<TokenType::ENDL>() ||
                    parse_lines<AstLine, parse_line>({ tarr, end }, indent_level + 1, ast_line.line_branch.body))
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
                if (parse_expr({ tarr, i + 1, end }, ast_line.line_branch.cond, indent_level))
                    return error::syntax(tarr[i], "Invalid syntax in elif condition");

                for (end++; end < tarr.size() && tarr[end]->ty == TokenType::INDENT; end++);
                if (end == tarr.size())
                    return error::syntax(tarr[(size_t)end - 1], "Missing body of evaluation mode block");
                if (tarr[end]->expect<TokenType::ENDL>() ||
                    parse_lines<AstLine, parse_line>({ tarr, end }, indent_level + 1, ast_line.line_branch.body))
                    return true;
                break;

            case TokenType::KW_ELSE:
                new (&ast_line.line_block) AstLineBlock();
                ast_line.ty = LineType::ELSE;

                for (end = i + 1; tarr[end]->is_whitespace(); end++);
                if (tarr[end]->expect<TokenType::COLON>())
                    return true;

                for (end++; end < tarr.size() && tarr[end]->ty == TokenType::INDENT; end++);
                if (end == tarr.size())
                    return error::syntax(tarr[(size_t)end - 1], "Missing body of evaluation mode block");
                if (tarr[end]->expect<TokenType::ENDL>() ||
                    parse_lines<AstLine, parse_line>({ tarr, end }, indent_level + 1, ast_line.line_block.body))
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
                if (parse_expr({ tarr, i + 1, end }, ast_line.line_branch.cond, indent_level))
                    return error::syntax(tarr[i], "Invalid syntax in while condition");

                for (end++; end < tarr.size() && tarr[end]->ty == TokenType::INDENT; end++);
                if (end == tarr.size())
                    return error::syntax(tarr[(size_t)end - 1], "Missing body of evaluation mode block");
                if (tarr[end]->expect<TokenType::ENDL>() ||
                    parse_lines<AstLine, parse_line>({ tarr, end }, indent_level + 1, ast_line.line_branch.body))
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
                if (parse_expr({ tarr, i + 1, end }, ast_line.line_for.decl, indent_level))
                    return error::syntax(tarr[i], "Invalid syntax in for loop counter");
                i = end;

                end = tarr.search(IsSameCriteria(TokenType::COLON), i);
                if (end == -1)
                    return error::syntax(tarr[i], "Missing ':' in for loop");
                if (end == i + 1)
                    return error::syntax(tarr[i], "Empty expression in for loop iterator");
                if (parse_expr({ tarr, i + 1, end }, ast_line.line_for.iter, indent_level))
                    return error::syntax(tarr[i], "Invalid syntax in for loop iterator");

                for (end++; end < tarr.size() && tarr[end]->ty == TokenType::INDENT; end++);
                if (end == tarr.size())
                    return error::syntax(tarr[(size_t)end - 1], "Missing body of evaluation mode block");
                if (tarr[end]->expect<TokenType::ENDL>() ||
                    parse_lines<AstLine, parse_line>({ tarr, end }, indent_level + 1, ast_line.line_for.body))
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
                    parse_lines<AstLine, parse_line>({ tarr, i }, indent_level + 1, ast_line.line_label.body))
                    return true;
                break;

            default:
                new (&ast_line.line_expr) AstLineExpr();
                ast_line.ty = LineType::EXPR;
                if (parse_expr({ tarr, i }, ast_line.line_expr.line, indent_level))
                    return true;
                break;
            }
            return false;
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
            return parse_lines<AstLine, parse_line>({ tarr, i }, indent_level + 1, ast_entry.lines);
        }

        bool parse_signature(const TokenArray& tarr, AstCargSig& ast_carg_sig, int indent_level)
        {
            assert(tarr.size() > 0);

            // Fast-forwarding to the start of the signature
            int i = 0;
            for (; tarr[i]->is_whitespace(); i++);
            if (i == tarr.size() || tarr[i]->ty != TokenType::ANGLE_O)
                return false;  // No cargs

            // Finding the end of the cargs
            int end = tarr.search(IsSameCriteria(TokenType::ANGLE_C), i = 1);
            if (end == -1)
                return error::syntax(tarr[i], "Unable to find closing bracket '>' for cargs signature");

            // Parsing out the cargs
            if (parse_args({ tarr, i + 1, end }, ast_carg_sig.cargs, indent_level + 1))
                return true;
            for (i = end + 1; i < tarr.size() && tarr[i]->is_whitespace(); i++);
            if (i != tarr.size())
                return error::syntax(tarr[i], "Unexpected token '%' after cargs in signature", tarr[i]);
            return false;
        }

        bool parse_signature(const TokenArray& tarr, AstFnSig& ast_fn_sig, int indent_level)
        {
            assert(tarr.size() > 0);

            // Fast-forwarding to the start of the signature
            int i = 0;
            for (; tarr[i]->is_whitespace(); i++);
            if (i == tarr.size())
                return error::syntax(tarr[tarr.size() - 1], "Unexpected end of function signature");

            // Parsing out the cargs from the fn definition (if there are any)
            if (tarr[i]->ty == TokenType::ANGLE_O)
            {
                // Finding the end of the cargs
                int end = tarr.search(IsSameCriteria(TokenType::ANGLE_C), i = 1);
                if (end == -1)
                    return error::syntax(tarr[i], "Unable to find closing bracket '>' for cargs in function signature");

                // Parsing out the cargs
                if (parse_args({ tarr, i + 1, end }, ast_fn_sig.cargs, indent_level + 1))
                    return true;
                for (i = end + 1; i < tarr.size() && tarr[i]->is_whitespace(); i++);
                if (i == tarr.size())
                    return error::syntax(tarr[i], "Unexpected end of signature after cargs in function signature");
            }

            // Parsing out the vargs from the fn definition
            if (tarr[i]->expect<TokenType::ROUND_O>())
                return true;

            // Finding the end of the vargs
            int end = tarr.search(IsSameCriteria(TokenType::ANGLE_C), i + 1);
            if (end == -1)
                return error::syntax(tarr[i], "Unable to find closing bracket ')' for vargs in function signature");

            // Parsing out the vargs
            if (parse_args({ tarr, i + 1, end }, ast_fn_sig.vargs, indent_level + 1))
                return true;
            for (i = end + 1; i < tarr.size() && tarr[i]->is_whitespace(); i++);

            // Checking for a return arrow
            if (i == tarr.size())
                return false;  // No return values

            // Parsing out the return values
            return
                tarr[i]->expect<TokenType::ARROW>() ||
                parse_args({ tarr, i + 1 }, ast_fn_sig.rets, indent_level + 1);
        }

        bool parse_signature(const TokenArray& tarr, AstBlockSig& ast_block_sig, int indent_level)
        {
            assert(tarr.size() > 0);

            // Fast-forwarding to the start of the signature
            int i = 0;
            for (; tarr[i]->is_whitespace(); i++);
            if (i == tarr.size())
                return error::syntax(tarr[tarr.size() - 1], "Unexpected end of function signature");

            // Parsing out the cargs from the fn definition (if there are any)
            if (tarr[i]->ty == TokenType::ANGLE_O)
            {
                // Finding the end of the cargs
                int end = tarr.search(IsSameCriteria(TokenType::ANGLE_C), i = 1);
                if (end == -1)
                    return error::syntax(tarr[i], "Unable to find closing bracket '>' for cargs in function signature");

                // Parsing out the cargs
                if (parse_args({ tarr, i + 1, end }, ast_block_sig.cargs, indent_level + 1))
                    return true;
                for (i = end + 1; i < tarr.size() && tarr[i]->is_whitespace(); i++);
                if (i == tarr.size())
                    return error::syntax(tarr[i], "Unexpected end of signature after cargs in function signature");
            }

            // Parsing out the vargs from the fn definition
            if (tarr[i]->expect<TokenType::ROUND_O>())
                return true;

            // Finding the end of the vargs
            int end = tarr.search(IsSameCriteria(TokenType::ANGLE_C), i + 1);
            if (end == -1)
                return error::syntax(tarr[i], "Unable to find closing bracket ')' for vargs in function signature");

            // Parsing out the vargs
            if (parse_args({ tarr, i + 1, end }, ast_block_sig.vargs, indent_level + 1))
                return true;
            for (i = end + 1; i < tarr.size() && tarr[i]->is_whitespace(); i++);

            // Checking for a return arrow
            if (i == tarr.size())
                return false;  // No return values

            // Parsing out the return values...  Yes, I am extremely lazy
            std::vector<AstExpr> rets;
            if (tarr[i]->expect<TokenType::ARROW>() ||
                parse_args({ tarr, i + 1 }, rets, indent_level + 1)
                ) return true;
            for (AstExpr& ret : rets)
            {
                if (ret.ty != AstExpr::Type::VAR)
                    return error::syntax(ret.node_info, "Expected only indentifiers in the returns of a block");
                ast_block_sig.rets.push_back(ret.expr_string);
            }
            return false;
        }

        int parse_unary(const TokenArray& tarr, AstExpr& ast_expr, int ilv)
        {
            int i;
            int end;
            switch (tarr[0]->ty)
            {
            case TokenType::KW_NAMESPACE:
                new (&ast_expr.expr_namespace) AstExprNamespace();
                ast_expr.ty = AstExpr::Type::DEFN_NAMESPACE;
                for (i = 1; i < tarr.size() && tarr[i]->is_whitespace(); i++);
                if (tarr[i]->expect<TokenType::COLON>())
                    return -1;
                end = tarr.search(LineEndCriteria(ilv), i + 1);
                if (end == -1)
                    end = tarr.size();
                if (parse_lines<AstExpr, parse_expr>({ tarr, i + 1, end }, ilv + 1, ast_expr.expr_namespace.lines))
                    return -1;
                return end;

            case TokenType::KW_STRUCT:
                new (&ast_expr.expr_struct) AstExprStruct();
                ast_expr.ty = AstExpr::Type::DEFN_STRUCT;
                i = tarr.search(IsSameCriteria(TokenType::COLON), 1);
                if (i == -1)
                {
                    error::syntax(tarr[0], "Unable to find a ':' to terminate struct signature");
                    return -1;
                }
                if (parse_signature({ tarr, i, end }, ast_expr.expr_struct.signature, ilv))
                    return -1;
                end = tarr.search(LineEndCriteria(ilv), i + 1);
                if (end == -1)
                    end = tarr.size();
                if (parse_lines<AstExpr, parse_expr>({ tarr, i + 1, end }, ilv + 1, ast_expr.expr_struct.body))
                    return -1;
                return end;

            case TokenType::KW_ENUM:
                new (&ast_expr.expr_enum) AstExprEnum();
                ast_expr.ty = AstExpr::Type::DEFN_ENUM;
                i = tarr.search(IsSameCriteria(TokenType::COLON), 1);
                if (i == -1)
                {
                    error::syntax(tarr[0], "Unable to find a ':' to terminate enum signature");
                    return -1;
                }
                if (parse_signature({ tarr, i, end }, ast_expr.expr_enum.signature, ilv))
                    return -1;
                end = tarr.search(LineEndCriteria(ilv), i + 1);
                if (end == -1)
                    end = tarr.size();
                if (parse_lines<AstEnumEntry, parse_enum_entry>({ tarr, i + 1, end }, ilv + 1, ast_expr.expr_enum.entries))
                    return -1;
                // Making sure that no labels are duplicated in the enum, and building out the entry map
                for (const auto& entry : ast_expr.expr_enum.entries)
                {
                    const auto& it = ast_expr.expr_enum.entry_map.find(entry.name);
                    if (it != ast_expr.expr_enum.entry_map.end())
                    {
                        error::syntax(it->second->node_info, "Found duplicate label % in enum definition", it->first);
                        end = -1;
                    }
                    else
                        ast_expr.expr_enum.entry_map[entry.name] = &entry;
                }
                return end;

            case TokenType::KW_DEF:
                i = tarr.search(IsInCriteria({ TokenType::COLON, TokenType::SIGDECL }), 1);
                if (i == -1)
                {
                    error::syntax(tarr[0], "Unable to find a ':' or ';' to terminate def signature");
                    return -1;
                }
                if (tarr[i]->ty == TokenType::SIGDECL)
                {
                    new (&ast_expr.expr_block_decl) AstBlockSig();
                    ast_expr.ty = AstExpr::Type::DECL_DEF;
                    if (parse_signature({ tarr, 1, i }, ast_expr.expr_block_decl, ilv))
                        return -1;
                    return i + 1;
                }
                new (&ast_expr.expr_block) AstExprBlock();
                ast_expr.ty = AstExpr::Type::DEFN_DEF;
                if (parse_signature({ tarr, 1, i }, ast_expr.expr_block.signature, ilv))
                    return -1;
                end = tarr.search(LineEndCriteria(ilv), i + 1);
                if (end == -1)
                    end = tarr.size();
                for (i++; i < end && tarr[i]->is_whitespace(); i++);
                if (i == end)
                {
                    // Empty body
                    ast_expr.expr_block.is_bytecode = false;
                }
                else if (tarr[i]->ty == TokenType::COLON)
                {
                    // bytecode
                    ast_expr.expr_block.is_bytecode = true;
                    ast_expr.expr_block.tarr = { tarr, i, end };
                }
                else
                {
                    ast_expr.expr_block.is_bytecode = false;
                    if (parse_lines<AstExpr, parse_expr>({ tarr, i, end }, ilv + 1, ast_expr.expr_block.body))
                        return -1;
                }
                return end;

            case TokenType::KW_INTR:
                i = tarr.search(IsInCriteria({ TokenType::COLON, TokenType::SIGDECL }), 1);
                if (i == -1)
                {
                    error::syntax(tarr[0], "Unable to find a ':' or ';' to terminate intr signature");
                    return -1;
                }
                if (tarr[i]->ty == TokenType::SIGDECL)
                {
                    new (&ast_expr.expr_block_decl) AstBlockSig();
                    ast_expr.ty = AstExpr::Type::DECL_INTR;
                    if (parse_signature({ tarr, 1, i }, ast_expr.expr_block_decl, ilv))
                        return -1;
                    return i + 1;
                }
                new (&ast_expr.expr_block) AstExprBlock();
                ast_expr.ty = AstExpr::Type::DEFN_INTR;
                if (parse_signature({ tarr, 1, i }, ast_expr.expr_block.signature, ilv))
                    return -1;
                end = tarr.search(LineEndCriteria(ilv), i + 1);
                if (end == -1)
                    end = tarr.size();
                for (i++; i < end && tarr[i]->is_whitespace(); i++);
                if (i == end)
                {
                    // Empty body
                    ast_expr.expr_block.is_bytecode = false;
                }
                else if (tarr[i]->ty == TokenType::COLON)
                {
                    // bytecode
                    ast_expr.expr_block.is_bytecode = true;
                    ast_expr.expr_block.tarr = { tarr, i, end };
                }
                else
                {
                    ast_expr.expr_block.is_bytecode = false;
                    if (parse_lines<AstExpr, parse_expr>({ tarr, i, end }, ilv + 1, ast_expr.expr_block.body))
                        return -1;
                }
                return end;

            case TokenType::KW_FN:
                i = tarr.search(IsInCriteria({ TokenType::COLON, TokenType::SIGDECL }), 1);
                if (i == -1)
                {
                    error::syntax(tarr[0], "Unable to find a ':' or ';' to terminate fn signature");
                    return -1;
                }
                if (tarr[i]->ty == TokenType::SIGDECL)
                {
                    new (&ast_expr.expr_fn_decl) AstFnSig();
                    ast_expr.ty = AstExpr::Type::DECL_FN;
                    if (parse_signature({ tarr, 1, i }, ast_expr.expr_fn_decl, ilv))
                        return -1;
                    return i + 1;
                }
                new (&ast_expr.expr_fn) AstExprFn();
                ast_expr.ty = AstExpr::Type::DEFN_FN;
                if (parse_signature({ tarr, 1, i }, ast_expr.expr_fn.signature, ilv))
                    return -1;
                end = tarr.search(LineEndCriteria(ilv), i + 1);
                if (end == -1)
                    end = tarr.size();
                for (i++; i < end && tarr[i]->is_whitespace(); i++);
                if (i == end)
                {
                    // Empty body
                    ast_expr.expr_fn.is_bytecode = false;
                }
                else if (tarr[i]->ty == TokenType::COLON)
                {
                    // bytecode
                    ast_expr.expr_fn.is_bytecode = true;
                    ast_expr.expr_fn.tarr = { tarr, i, end };
                }
                else
                {
                    ast_expr.expr_fn.is_bytecode = false;
                    if (parse_lines<AstExpr, parse_expr>({ tarr, i, end }, ilv + 1, ast_expr.expr_fn.body))
                        return -1;
                }
                return end;

            case TokenType::KW_INIT:
                i = tarr.search(IsInCriteria({ TokenType::COLON, TokenType::SIGDECL }), 1);
                if (i == -1)
                {
                    error::syntax(tarr[0], "Unable to find a ':' or ';' to terminate init signature");
                    return -1;
                }
                if (tarr[i]->ty == TokenType::SIGDECL)
                {
                    new (&ast_expr.expr_init_decl) AstCargSig();
                    ast_expr.ty = AstExpr::Type::DECL_INIT;
                    if (parse_signature({ tarr, 1, i }, ast_expr.expr_init_decl, ilv))
                        return -1;
                    return i + 1;
                }
                new (&ast_expr.expr_init) AstExprInit();
                ast_expr.ty = AstExpr::Type::DEFN_INIT;
                if (parse_signature({ tarr, 1, i }, ast_expr.expr_init.signature, ilv))
                    return -1;
                end = tarr.search(LineEndCriteria(ilv), i + 1);
                if (end == -1)
                    end = tarr.size();
                for (i++; i < end && tarr[i]->is_whitespace(); i++);
                if (i == end)
                {
                    // Empty body
                    ast_expr.expr_init.is_bytecode = false;
                }
                else if (tarr[i]->ty == TokenType::COLON)
                {
                    // bytecode
                    ast_expr.expr_init.is_bytecode = true;
                    ast_expr.expr_init.tarr = { tarr, i, end };
                }
                else
                {
                    ast_expr.expr_init.is_bytecode = false;
                    if (parse_lines<AstExpr, parse_expr>({ tarr, i, end }, ilv + 1, ast_expr.expr_init.body))
                        return -1;
                }
                return end;

            case TokenType::KW_IMPORT:
                new (&ast_expr.expr_import) AstExprImport();
                ast_expr.ty = AstExpr::Type::IMPORT;

                for (end = i + 1; end < tarr.size() && tarr[end]->is_whitespace(); end++);
                if (end == tarr.size())
                    return error::syntax(tarr[i], "Expected identifier after 'import'");
                if (tarr[end]->expect<TokenType::IDN>())
                    return true;
                ast_expr.expr_import.imp.push_back(tarr[end]->get<TokenType::IDN>().val);
                for (end++; end < tarr.size() && tarr[end]->is_whitespace(); end++);
                while (end < tarr.size() && tarr[end]->ty == TokenType::DOT)
                {
                    if (tarr[end]->expect<TokenType::DOT>())
                        return true;
                    for (end++; end < tarr.size() && tarr[end]->is_whitespace(); end++);
                    if (end == tarr.size())
                        return error::syntax(tarr[i], "Expected identifier after '.' in import statement");
                    if (tarr[end]->expect<TokenType::IDN>())
                        return true;
                    ast_expr.expr_import.imp.push_back(tarr[end]->get<TokenType::IDN>().val);
                    for (end++; end < tarr.size() && tarr[end]->is_whitespace(); end++);
                }
                return end;

            case TokenType::ADD:
                ast_expr.ty = AstExpr::Type::UNARY_POS;
                goto parse_unary_op;
            case TokenType::SUB:
                ast_expr.ty = AstExpr::Type::UNARY_NEG;
                goto parse_unary_op;
            case TokenType::STAR:
                ast_expr.ty = AstExpr::Type::UNARY_UNPACK;
                goto parse_unary_op;
            case TokenType::KW_NOT:
                ast_expr.ty = AstExpr::Type::UNARY_NOT;
                goto parse_unary_op;
            case TokenType::KW_MUT:
                ast_expr.ty = AstExpr::Type::UNARY_MUT;
                goto parse_unary_op;
            case TokenType::KW_REF:
                ast_expr.ty = AstExpr::Type::UNARY_REF;
                goto parse_unary_op;
            parse_unary_op:
                new (&ast_expr.expr_unary) AstExprUnaryOp();
                i = tarr.search(PrioritySplitCriteria<default_prec_table>(), 1);
                if (i == -1)
                    i = tarr.size();
                if (parse_leaf_expr({ tarr, 1, i }, ast_expr, ilv))
                    return -1;
                return i;
            }

            error::syntax(tarr[0], "Unexpected token '%' for the start of an expression", tarr[0]);
            return -1;
        }

        template<PrecTable prec_table, RtolTable rtol_table>
        int parse_subexpr(const TokenArray& tarr, AstExpr& ast_expr, int ilv, int prec)
        {
            bool tuple_lit = tarr[0]->ty == TokenType::COMMA;
            if (tuple_lit)
            {
                AstExpr tmp = std::move(ast_expr);
                new (&ast_expr.expr_agg) AstExprAggLit();
                ast_expr.node_info = tmp.node_info;
                tmp.node_info.line_end = tarr[0]->line_num;
                tmp.node_info.col_end = tarr[0]->col_num;
                ast_expr.expr_agg.elems.push_back(std::move(tmp));
            }

            int end = 0;
            int i;
            int op_prec = prec_table[(size_t)tarr[end]->ty];
            while (op_prec >= prec)
            {
                AstExpr::Type binop_type = token_expr_table[(size_t)tarr[end]->ty];
                for (i = end + 1; i < tarr.size() && tarr[i]->is_whitespace(); i++);
                
                AstExpr rhs;
                rhs.node_info = {  // The start info for the expression
                    .fname = tarr[end]->fname,
                    .line_start = tarr[end]->line_num,
                    .col_start = tarr[end]->col_num,
                };
                if (i == tarr.size())
                {
                    // Empty expression
                    if (tuple_lit)
                        return i;  // Don't worry about empty types for tuples, commas at the end are allowed
                    rhs.ty = AstExpr::Type::EMPTY;
                    rhs.node_info.line_end = rhs.node_info.line_start;
                    rhs.node_info.col_end = rhs.node_info.col_start;
                }
                else
                {
                    end = tarr.search(PrioritySplitCriteria(), i);
                    if (i == end)
                    {
                        // Either a unary operator or a code block
                        end = parse_unary({ tarr, i }, rhs, ilv);
                        if (end == -1) return -1;
                        for (; end < tarr.size() && tarr[end]->is_whitespace(); end++);
                    }
                    else
                    {
                        // Nothin special, just a regular old binary op
                        if (end == -1)
                            end = tarr.size();
                        if (parse_leaf_expr({ tarr, i, end }, rhs, ilv))
                            return -1;
                    }
                }

                if (end == tarr.size())
                {
                    // Reached the end of the expression
                    rhs.node_info.line_end = tarr[end - 1]->line_num;
                    rhs.node_info.col_end = tarr[end - 1]->col_num;
                    if (tuple_lit)
                    {
                        ast_expr.expr_agg.elems.push_back(std::move(rhs));
                        return tarr.size();
                    }
                    else
                    {
                        AstExpr lhs = std::move(ast_expr);  // lhs <- ast_expr; ast_expr <- AstExpr();
                        new (&ast_expr.expr_binary) AstExprBinaryOp();
                        ast_expr.ty = token_expr_table[(size_t)tarr[end]->ty];
                        ast_expr.node_info = lhs.node_info;
                        lhs.node_info.line_end = tarr[i].line_num;
                        lhs.node_info.col_end = tarr[i].col_num;
                        ast_expr.expr_binary.left = std::make_unique<AstExpr>(std::move(lhs));
                        ast_expr.expr_binary.right = std::make_unique<AstExpr>(std::move(rhs));
                        return tarr.size();
                    }
                }
                rhs.node_info.line_end = tarr[end]->line_num;
                rhs.node_info.col_end = tarr[end]->col_num;

                i = end;  // Keeping track of the end of the left hand side
                int split_prec = prec_table[(size_t)tarr[end]->ty];
                while (split_prec > op_prec || (split_prec == op_prec && rtol_table[split_prec]))
                {
                    // Either the precedence increased, or we're dealing with right associativity.
                    // In both cases the right side need to be pushed down, so recurse on it
                    end = parse_subexpr<prec_table, rtol_table>({ tarr, end }, rhs, ilv, op_prec + 1);
                    if (end == -1)
                        return -1;
                    if (end == tarr.size())
                        break;
                    split_prec = prec_table[(size_t)tarr[end]->ty];
                }
                
                // Same precedence with left associativity,
                // push down the left side of the tree and continue looping
                if (tuple_lit)
                {
                    ast_expr.expr_agg.elems.push_back(std::move(rhs));
                }
                else
                {
                    AstExpr lhs = std::move(ast_expr);  // lhs <- ast_expr; ast_expr <- AstExpr();
                    new (&ast_expr.expr_binary) AstExprBinaryOp();
                    ast_expr.ty = binop_type;
                    ast_expr.node_info = lhs.node_info;
                    lhs.node_info.line_end = tarr[i].line_num;
                    lhs.node_info.col_end = tarr[i].col_num;
                    ast_expr.expr_binary.left = std::make_unique<AstExpr>(std::move(lhs));
                    ast_expr.expr_binary.right = std::make_unique<AstExpr>(std::move(rhs));
                }
                
                if (end == tarr.size())
                    return tarr.size();
                op_prec = prec_table[(size_t)tarr[end]->ty];
            }
            // priority dropped, so return the last token parsed
            return end;
        }

        template<PrecTable prec_table, RtolTable rtol_table>
        bool parse_expr(const TokenArray& tarr, AstExpr& ast_expr, int ilv)
        {
            assert(tarr.size() > 0);
            ast_expr.node_info = {
                .fname = tarr[0]->fname,
                .line_start = tarr[0]->line_num,
                .line_end = tarr[tarr.size() - 1]->line_num,
                .col_start = tarr[0]->col_num,
                .col_end = tarr[tarr.size() - 1]->col_num
            };

            // Parsing out the first expression
            int i = 0;
            for (; i < tarr.size() && tarr[i]->is_whitespace(); i++);
            if (i == tarr.size())
                return error::syntax(tarr[0], "Empty expression are not allowed");
            int end = tarr.search(PrioritySplitCriteria(), i);

            if (i == end)
            {
                // Either a unary operator or a code block
                end = parse_unary({ tarr, i }, ast_expr, ilv);
                if (end == -1) return -1;
                for (; end < tarr.size() && tarr[end]->is_whitespace(); end++);
            }
            else
            {
                // Nothin special, just a regular old binary op
                if (end == -1)
                    end = tarr.size();
                if (parse_leaf_expr({ tarr, i, end }, ast_expr, ilv))
                    return -1;
            }
            if (end == tarr.size())
                return false;

            end = parse_subexpr<prec_table, rtol_table>({ tarr, end }, ast_expr, ilv, 0);
            if (end == -1)
                return true;
            if (end < tarr.size())  // I don't think this should every happen, idk though
                return error::syntax(tarr[end], "Unexpected token '%'", tarr[end]);
            return false;
        }

        bool parse_module(const TokenArray& tarr, AstModule& ast_module)
        {
            ast_module.fname = tarr[0]->fname;
            return parse_lines<AstExpr, parse_expr>(tarr, 0, ast_module.lines);
        }

        AstExpr::AstExpr() {}

        AstExpr::AstExpr(AstExpr&& expr) noexcept
        {
            node_info = std::move(expr.node_info);

#define DO_MOVE(idn) new (&idn) decltype(idn)(std::move(expr.idn))
            switch (ty)
            {
            case Type::INVALID:
            case Type::EMPTY:
                break;
            case Type::KW:
                DO_MOVE(expr_kw);
                break;
            case Type::VAR:
                DO_MOVE(expr_string);
                break;
            case Type::LIT_BOOL:
            case Type::LIT_INT:
            case Type::LIT_FLOAT:
                break;
            case Type::LIT_STRING:
                DO_MOVE(expr_string);
                break;
            case Type::LIT_ARRAY:
            case Type::LIT_TUPLE:
                DO_MOVE(expr_agg);
                break;
            case Type::UNARY_POS:
            case Type::UNARY_NEG:
            case Type::UNARY_NOT:
            case Type::UNARY_UNPACK:
            case Type::UNARY_MUT:
            case Type::UNARY_REF:
                DO_MOVE(expr_unary);
                break;
            case Type::BINARY_ADD:
            case Type::BINARY_SUB:
            case Type::BINARY_MUL:
            case Type::BINARY_DIV:
            case Type::BINARY_MOD:
            case Type::BINARY_POW:
            case Type::BINARY_IADD:
            case Type::BINARY_ISUB:
            case Type::BINARY_IMUL:
            case Type::BINARY_IDIV:
            case Type::BINARY_IMOD:
            case Type::BINARY_IPOW:
            case Type::BINARY_ASSIGN:
            case Type::BINARY_AND:
            case Type::BINARY_OR:
            case Type::BINARY_CMP_EQ:
            case Type::BINARY_CMP_NE:
            case Type::BINARY_CMP_GT:
            case Type::BINARY_CMP_LT:
            case Type::BINARY_CMP_GE:
            case Type::BINARY_CMP_LE:
            case Type::BINARY_CAST:
            case Type::BINARY_DECL:
                DO_MOVE(expr_binary);
                break;
            case Type::INDEX:
                DO_MOVE(expr_index);
                break;
            case Type::DOT:
                DO_MOVE(expr_name);
                break;
            case Type::CALL_CARGS:
            case Type::CALL_VARGS:
                DO_MOVE(expr_call);
                break;
            case Type::DEFN_NAMESPACE:
                DO_MOVE(expr_namespace);
                break;
            case Type::DEFN_ENUM:
                DO_MOVE(expr_enum);
                break;
            case Type::DEFN_STRUCT:
                DO_MOVE(expr_struct);
                break;
            case Type::DEFN_DEF:
            case Type::DEFN_INTR:
                DO_MOVE(expr_block);
                break;
            case Type::DEFN_FN:
                DO_MOVE(expr_fn);
                break;
            case Type::DEFN_INIT:
                DO_MOVE(expr_init);
                break;
            case Type::DECL_DEF:
            case Type::DECL_INTR:
                DO_MOVE(expr_block_decl);
                break;
            case Type::DECL_FN:
                DO_MOVE(expr_fn_decl);
                break;
            case Type::DECL_INIT:
                DO_MOVE(expr_init_decl);
                break;
            case Type::IMPORT:
                DO_MOVE(expr_import);
                break;
#undef DO_MOVE
            default:
                assert(false);
            }
            expr.reset();
        }

        AstExpr& AstExpr::operator=(AstExpr&& expr) noexcept
        {
            if (this == &expr)
                return *this;
            this->~AstExpr();
            new (this) AstExpr(std::move(expr));
            return *this;
        }

        AstExpr::~AstExpr()
        {
            reset();
        }

        void AstExpr::reset() noexcept
        {
            node_info = { "", 0, 0, 0, 0 };

            switch (ty)
            {
            case Type::INVALID:
            case Type::EMPTY:
                break;
            case Type::KW:
                expr_kw.~ExprKW();
                break;
            case Type::VAR:
                expr_string.~basic_string();
                break;
            case Type::LIT_BOOL:
            case Type::LIT_INT:
            case Type::LIT_FLOAT:
                break;
            case Type::LIT_STRING:
                expr_string.~basic_string();
                break;
            case Type::LIT_ARRAY:
                expr_agg.~AstExprAggLit();
                break;
            case Type::LIT_TUPLE:
                expr_agg.~AstExprAggLit();
                break;
            case Type::UNARY_POS:
            case Type::UNARY_NEG:
            case Type::UNARY_NOT:
            case Type::UNARY_UNPACK:
            case Type::UNARY_MUT:
            case Type::UNARY_REF:
                expr_unary.~AstExprUnaryOp();
                break;
            case Type::BINARY_ADD:
            case Type::BINARY_SUB:
            case Type::BINARY_MUL:
            case Type::BINARY_DIV:
            case Type::BINARY_MOD:
            case Type::BINARY_POW:
            case Type::BINARY_IADD:
            case Type::BINARY_ISUB:
            case Type::BINARY_IMUL:
            case Type::BINARY_IDIV:
            case Type::BINARY_IMOD:
            case Type::BINARY_IPOW:
            case Type::BINARY_ASSIGN:
            case Type::BINARY_AND:
            case Type::BINARY_OR:
            case Type::BINARY_CMP_EQ:
            case Type::BINARY_CMP_NE:
            case Type::BINARY_CMP_GT:
            case Type::BINARY_CMP_LT:
            case Type::BINARY_CMP_GE:
            case Type::BINARY_CMP_LE:
            case Type::BINARY_CAST:
            case Type::BINARY_DECL:
                expr_binary.~AstExprBinaryOp();
                break;
            case Type::INDEX:
                expr_index.~AstExprIndex();
                break;
            case Type::DOT:
                expr_name.~AstExprName();
                break;
            case Type::CALL_CARGS:
            case Type::CALL_VARGS:
                expr_call.~AstExprCall();
                break;
            case Type::DEFN_NAMESPACE:
                expr_namespace.~AstExprNamespace();
                break;
            case Type::DEFN_ENUM:
                expr_enum.~AstExprEnum();
                break;
            case Type::DEFN_STRUCT:
                expr_struct.~AstExprStruct();
                break;
            case Type::DEFN_DEF:
            case Type::DEFN_INTR:
                expr_block.~AstExprBlock();
                break;
            case Type::DEFN_FN:
                expr_fn.~AstExprFn();
                break;
            case Type::DEFN_INIT:
                expr_init.~AstExprInit();
                break;
            case Type::DECL_DEF:
            case Type::DECL_INTR:
                expr_block_decl.~AstBlockSig();
                break;
            case Type::DECL_FN:
                expr_fn_decl.~AstFnSig();
                break;
            case Type::DECL_INIT:
                expr_init_decl.~AstCargSig();
                break;
            case Type::IMPORT:
                expr_import.~AstExprImport();
                break;
            default:
                assert(false);
            }
            ty = Type::INVALID;
        }
    }
}
