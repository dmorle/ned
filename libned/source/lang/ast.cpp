#include <ned/lang/ast.h>
#include <ned/lang/obj.h>

#include <functional>
#include <cassert>

using namespace nn::lang;

// Parses comma deliminated token sequences within a matching set of brackets
// Returns the index in tarr immediately after the CLOSE token was found
// On failure, this function returns -1
template<typename T, bool(*parse_fn)(Errors&, const TokenArray&, T&), TokenType OPEN, TokenType CLOSE>
inline int parse_args(Errors& errs, const TokenArray& tarr, int i, std::vector<T>& args)
{
    assert(tarr[0]->ty == OPEN);
    
    // Advancing one token past the opening bracket
    i++;
    int end = tarr.search(ArgEndCriteria(CLOSE), i);
    if (tarr[end]->ty == CLOSE)
    {
        // either a single arg, or its empty
        for (int j = i; j < end; j++)
            if (!tarr[j]->is_whitespace())
            {
                // single arg, parse it and move on
                args.push_back(T());
                if (parse_fn(errs, { tarr, i, end }, args.back()))
                {
                    args.clear();
                    return -1;
                }
                break;
            }
        return end;
    }

    // General case for parsing multiple args
    args.push_back(T());
    if (parse_fn(errs, { tarr, i, end }, args.back()))
    {
        args.clear();
        return -1;
    }
    do
    {
        end = tarr.search(ArgEndCriteria(CLOSE), i);
        args.push_back(T());
        if (parse_fn(errs, { tarr, i, end }, args.back()))
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
template<typename T, bool(*parse_fn)(Errors&, const TokenArray&, T&, int)>
inline bool parse_lines(Errors& errs, const TokenArray& tarr, int i, int indent_level, std::vector<T>& lines)
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
                if (parse_fn(errs, { tarr, i, end }, lines.back(), indent_level))
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

enum class OpAssoc
{
    LTOR,
    RTOL
};

// Eventually this can become constexpr, but it seems my current c++ compiler doesn't have that yet...
const std::vector<std::tuple<OpAssoc, std::vector<TokenType>>> prec_ops = {
    { OpAssoc::LTOR, { TokenType::COMMA }},
    { OpAssoc::RTOL, { TokenType::ASSIGN, TokenType::IADD, TokenType::ISUB, TokenType::IMUL, TokenType::IDIV, TokenType::IMOD }},
    { OpAssoc::LTOR, { TokenType::KW_AND, TokenType::KW_OR }},
    { OpAssoc::LTOR, { TokenType::CMP_EQ, TokenType::CMP_NE, TokenType::CMP_GT, TokenType::CMP_LT, TokenType::CMP_GE, TokenType::CMP_LE }},
    { OpAssoc::LTOR, { TokenType::ADD, TokenType::SUB }},
    { OpAssoc::LTOR, { TokenType::STAR, TokenType::DIV, TokenType::MOD }}
};

bool parse_leaf_mods(Errors& errs, const TokenArray& tarr, AstExpr& ast_expr, std::unique_ptr<AstExpr> lhs)
{
    return errs.add(tarr[0], "Parsing leaf mods has not been implemented yet");
}

bool parse_leaf_token(Errors& errs, const Token* ptk, AstExpr& ast_expr)
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
    case TokenType::KW_TYPE:
        ast_expr.expr_kw = ExprKW::TYPE;
        ast_expr.ty = ExprType::KW;
        return false;
    case TokenType::KW_VAR:
        ast_expr.expr_kw = ExprKW::VAR;
        ast_expr.ty = ExprType::KW;
        return false;
    case TokenType::KW_FP:
        ast_expr.expr_kw = ExprKW::FP;
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
        errs.add(ptk, "Unexpected token for single token expression leaf node");
        return true;
    }
}

template<int prec>
inline ExprType get_expr_type(TokenType ty);

template<>
inline ExprType get_expr_type<1>(TokenType ty)
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

template<int prec>
bool parse_expr_prec(Errors& errs, const TokenArray& tarr, AstExpr& ast_expr);

// specializations
template<> bool parse_expr_prec<7>(Errors& errs, const TokenArray& tarr, AstExpr& ast_expr);
template<> bool parse_expr_prec<6>(Errors& errs, const TokenArray& tarr, AstExpr& ast_expr);
template<> bool parse_expr_prec<0>(Errors& errs, const TokenArray& tarr, AstExpr& ast_expr);

template<int prec>
bool parse_expr_prec(Errors& errs, const TokenArray& tarr, AstExpr& ast_expr)
{
    static_assert(0 <= prec && prec <= 7);
    assert(tarr.size() > 0);
    const auto& [op_assoc, tys] = prec_ops[prec];  // TODO: update my c++ compiler and make this constexpr
    int pos;
    if (op_assoc == OpAssoc::LTOR)
        pos = tarr.search(IsInCriteria(tys), 1, -1);
    else
        pos = tarr.rsearch(IsInCriteria(tys));
    if (pos == -1)
        return parse_expr_prec<prec + 1>(errs, tarr, ast_expr);
    if (pos < 1 || tarr.size() - pos < 2)
        return errs.add(tarr[pos], "Dangling operator");
    std::unique_ptr<AstExpr> pleft = std::make_unique<AstExpr>();
    if (parse_expr_prec<prec + 1>(errs, { tarr, 0, pos }, *pleft))
        return true;
    std::unique_ptr<AstExpr> pright = std::make_unique<AstExpr>();
    if (parse_expr_prec<prec + 1>(errs, { tarr, pos + 1 }, *pright))
        return true;
    new (&ast_expr.expr_binary) AstExprBinaryOp();
    ast_expr.ty = get_expr_type<prec>(tarr[pos]->ty);
    ast_expr.expr_binary.left = std::move(pleft);
    ast_expr.expr_binary.right = std::move(pright);
    return false;
}

template<>
bool parse_expr_prec<7>(Errors& errs, const TokenArray& tarr, AstExpr& ast_expr)
{
    assert(tarr.size() > 0);

    if (tarr.size() == 1)  // single token leaf node
        return parse_leaf_token(errs, tarr[0], ast_expr);

    std::unique_ptr<AstExpr> lhs = nullptr;
    int i;
    if (tarr[0]->ty == TokenType::ROUND_O)
    {
        // handling bracketed expressions
        i = tarr.search(IsSameCriteria(TokenType::ROUND_C));
        if (i == -1)
            return errs.add(tarr[0], "Missing closing ')'");
        if (i == 1)
            return errs.add(tarr[0], "Empty expression are not allowed");

        // Checking if all the remaining tokens are whitespace
        // This is used to determine if any mods exist on the expression
        int end;
        for (end = i + 1; end < tarr.size() && tarr[end]->is_whitespace(); end++);
        if (end == tarr.size() - 1)
            return parse_expr(errs, { tarr, 1, i }, ast_expr);

        lhs = std::make_unique<AstExpr>();
        if (parse_expr(errs, { tarr, 1, i }, *lhs))
            return true;
        i = end;
    }
    else if (tarr[0]->ty == TokenType::SQUARE_O)
    {
        // handling array literals
        i = tarr.search(IsSameCriteria(TokenType::SQUARE_C));
        if (i == -1)
            return errs.add(tarr[0], "Missing closing ']'");
        
        // Checking if all the remaining tokens are whitespace
        // This is used to determine if any mods exist on the array literal
        int end;
        for (end = i + 1; end < tarr.size() && tarr[end]->is_whitespace(); end++);
        
        if (i == 1)
        {
            // Empty array
            if (end != tarr.size() - 1)
            {
                // has mods => initialize lhs and continue execution
                lhs = std::make_unique<AstExpr>();
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
            if (parse_expr(errs, { tarr, 1, i }, arr_expr))
                return true;
            if (end != tarr.size() - 1)
            {
                // has mods => initialize lhs and continue execution
                lhs = std::make_unique<AstExpr>();
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
    else if (tarr[0]->ty == TokenType::KW_FN)
    {
        // Function reference declaration
        return errs.add(tarr[0], "Function reference parsing has not been implemented");
    }
    else if (tarr[0]->ty == TokenType::KW_DEF)
    {
        // Block reference declaration
        return errs.add(tarr[0], "Block reference parsing has not been implemented");
    }
    else
    {
        // single token followed by modifiers
        lhs = std::make_unique<AstExpr>();
        if (parse_leaf_token(errs, tarr[0], *lhs))
            return true;
    }

    // If execution falls through the above code,
    // it means lhs and i are initialized and there are leaf mods that need to be parsed
    // i should be the index immediately after the lhs expression
    return parse_leaf_mods(errs, { tarr, i }, ast_expr, std::move(lhs));
}

template<>
bool parse_expr_prec<6>(Errors& errs, const TokenArray& tarr, AstExpr& ast_expr)
{
    assert(tarr.size() > 0);
    constexpr int prec = 6;

    // Handling unary operators
    std::unique_ptr<AstExpr> pexpr = nullptr;
    switch (tarr[0]->ty)
    {
        // Don't move repeated code out.  In the case where the type falls through,
        // keeping the code inside each case will save logic and a malloc
    case TokenType::ADD:
        if (tarr.size() < 2)
            return errs.add(tarr[0], "Dangling operator");
        pexpr = std::make_unique<AstExpr>();
        if (parse_expr_prec<prec + 1>(errs, { tarr, 1 }, *pexpr))
            return true;
        new (&ast_expr.expr_unary) AstExprUnaryOp();
        ast_expr.ty = ExprType::UNARY_POS;
        ast_expr.expr_unary.expr = std::move(pexpr);
        return false;
    case TokenType::SUB:
        if (tarr.size() < 2)
            return errs.add(tarr[0], "Dangling operator");
        pexpr = std::make_unique<AstExpr>();
        if (parse_expr_prec<prec + 1>(errs, { tarr, 1 }, *pexpr))
            return true;
        new (&ast_expr.expr_unary) AstExprUnaryOp();
        ast_expr.ty = ExprType::UNARY_NEG;
        ast_expr.expr_unary.expr = std::move(pexpr);
        return false;
    case TokenType::STAR:
        if (tarr.size() < 2)
            return errs.add(tarr[0], "Dangling operator");
        pexpr = std::make_unique<AstExpr>();
        if (parse_expr_prec<prec + 1>(errs, { tarr, 1 }, *pexpr))
            return true;
        new (&ast_expr.expr_unary) AstExprUnaryOp();
        ast_expr.ty = ExprType::UNARY_UNPACK;
        ast_expr.expr_unary.expr = std::move(pexpr);
        return false;
    case TokenType::KW_NOT:
        if (tarr.size() < 2)
            return errs.add(tarr[0], "Dangling operator");
        pexpr = std::make_unique<AstExpr>();
        if (parse_expr_prec<prec + 1>(errs, { tarr, 1 }, *pexpr))
            return true;
        new (&ast_expr.expr_unary) AstExprUnaryOp();
        ast_expr.ty = ExprType::UNARY_NOT;
        ast_expr.expr_unary.expr = std::move(pexpr);
        return false;
    }

    return parse_expr_prec<7>(errs, tarr, ast_expr);
}

template<>
bool parse_expr_prec<0>(Errors& errs, const TokenArray& tarr, AstExpr& ast_expr)
{
    assert(tarr.size() > 0);
    
    // Handling the comma operator
    IsSameCriteria sc{ TokenType::COMMA };
    int i = 0;
    int end;
    std::vector<AstExpr> elem_exprs;
    while ((end = tarr.search(sc, i)) > 0)
    {
        AstExpr elem_expr;
        if (parse_expr_prec<1>(errs, { tarr, i, end }, elem_expr))
            return true;
        elem_exprs.push_back(std::move(elem_expr));
        i = end + 1;
    }
    
    if (i == 0)  // No commas were found
        return parse_expr_prec<1>(errs, tarr, ast_expr);
    
    new (&ast_expr.expr_agg) AstExprAggLit();
    ast_expr.ty = ExprType::LIT_TUPLE;
    ast_expr.expr_agg.elems = std::move(elem_exprs);
    if (i == tarr.size())
        return false;  // In case the tuple was ended with a comma
    AstExpr last_expr;
    if (parse_expr_prec<1>(errs, { tarr, i }, last_expr))
        return true;
    ast_expr.expr_agg.elems.push_back(std::move(last_expr));
    return false;
}

namespace nn
{
    namespace lang
    {
        bool parse_expr(Errors& errs, const TokenArray& tarr, AstExpr& ast_expr)
        {
            assert(tarr.size() > 0);
            ast_expr.fname = tarr[0]->fname;
            ast_expr.line_start = tarr[0]->line_num;
            ast_expr.line_end = tarr[tarr.size() - 1]->line_num;
            ast_expr.col_start = tarr[0]->col_num;
            ast_expr.col_end = tarr[tarr.size() - 1]->col_num;
            
            int i = 0;
            for (; i < tarr.size() && tarr[i]->is_whitespace(); i++);
            if (i == tarr.size())
                return errs.add(tarr[0], "Empty expression are not allowed");
            int end = tarr.size() - 1;
            for (; tarr[end]->is_whitespace(); end--);
            return parse_expr_prec<0>(errs, { tarr, i, end + 1 }, ast_expr);
        }

        bool parse_line(Errors& errs, const TokenArray& tarr, AstLine& ast_line, int indent_level)
        {
            assert(tarr.size() > 0);
            ast_line.fname = tarr[0]->fname;
            ast_line.line_start = tarr[0]->line_num;
            ast_line.line_end = tarr[tarr.size() - 1]->line_num;
            ast_line.col_start = tarr[0]->col_num;
            ast_line.col_end = tarr[tarr.size() - 1]->col_num;

            int i = 0;
            for (; i < tarr.size() && tarr[i]->is_whitespace(); i++);
            if (i == tarr.size())
                return errs.add(tarr[0], "Empty lines are not allowed");
            int end;
            switch (tarr[i]->ty)
            {
            case TokenType::KW_BREAK:
                for (const auto& tk : TokenArray{ tarr, 1 })
                    if (!tk.is_whitespace())
                        return errs.add(tarr[0], "Unexpected token in break statement");
                ast_line.ty = LineType::BREAK;
                break;

            case TokenType::KW_CONTINUE:
                for (const auto& tk : TokenArray{ tarr, 1 })
                    if (!tk.is_whitespace())
                        return errs.add(tarr[0], "Unexpected token in continue statement");
                ast_line.ty = LineType::CONTINUE;
                break;

            case TokenType::KW_EXPORT:
                for (i++; tarr[i]->is_whitespace(); i++);
                if (tarr[i]->expect<TokenType::IDN>(errs))
                    return true;
                new (&ast_line.line_export) AstLineExport();
                ast_line.ty = LineType::EXPORT;
                ast_line.line_export.var_name = tarr[i]->get<TokenType::IDN>().val;
                break;

            case TokenType::KW_RAISE:
                new (&ast_line.line_func) AstLineUnaryFunc();
                ast_line.ty = LineType::RAISE;
                if (parse_expr(errs, { tarr, i + 1 }, ast_line.line_func.expr))
                    return errs.add(tarr[i], "Invalid syntax in raise statement");
                break;

            case TokenType::KW_RETURN:
                new (&ast_line.line_func) AstLineUnaryFunc();
                ast_line.ty = LineType::RETURN;
                if (parse_expr(errs, { tarr, i + 1 }, ast_line.line_func.expr))
                    return errs.add(tarr[i], "Invalid syntax in return statement");
                break;

            case TokenType::KW_PRINT:
                new (&ast_line.line_func) AstLineUnaryFunc();
                ast_line.ty = LineType::PRINT;
                if (parse_expr(errs, { tarr, i + 1 }, ast_line.line_func.expr))
                    return errs.add(tarr[i], "Invalid syntax in print statement");
                break;

            case TokenType::KW_IF:
                new (&ast_line.line_branch) AstLineBranch();
                ast_line.ty = LineType::IF;

                end = tarr.search(IsSameCriteria(TokenType::COLON));
                if (end == -1)
                    return errs.add(tarr[i], "Missing ':' in if statement");
                if (end == i + 1)
                    return errs.add(tarr[i], "Empty expression in if condition");
                if (parse_expr(errs, { tarr, i + 1, end }, ast_line.line_branch.cond))
                    return errs.add(tarr[i], "Invalid syntax in if condition");

                for (end++; tarr[end]->ty == TokenType::INDENT; end++);
                if (tarr[end]->expect<TokenType::ENDL>(errs) ||
                    parse_lines<AstLine, parse_line>(errs, tarr, end, indent_level + 1, ast_line.line_branch.body))
                    return true;
                break;

            case TokenType::KW_ELIF:
                new (&ast_line.line_branch) AstLineBranch();
                ast_line.ty = LineType::ELIF;

                end = tarr.search(IsSameCriteria(TokenType::COLON));
                if (end == -1)
                    return errs.add(tarr[i], "Missing ':' in elif statement");
                if (end == i + 1)
                    return errs.add(tarr[i], "Empty expression in elif condition");
                if (parse_expr(errs, { tarr, i + 1, end }, ast_line.line_branch.cond))
                    return errs.add(tarr[i], "Invalid syntax in elif condition");

                for (end++; tarr[end]->ty == TokenType::INDENT; end++);
                if (tarr[end]->expect<TokenType::ENDL>(errs) ||
                    parse_lines<AstLine, parse_line>(errs, tarr, end, indent_level + 1, ast_line.line_branch.body))
                    return true;
                break;

            case TokenType::KW_ELSE:
                new (&ast_line.line_else) AstLineElse();
                ast_line.ty = LineType::ELSE;
                
                for (end = i + 1; tarr[i]->is_whitespace(); i++);
                if (tarr[end]->expect<TokenType::COLON>(errs))
                    return true;

                for (end++; tarr[end]->ty == TokenType::INDENT; end++);
                if (tarr[end]->expect<TokenType::ENDL>(errs) ||
                    parse_lines<AstLine, parse_line>(errs, tarr, end, indent_level + 1, ast_line.line_else.body))
                    return true;
                break;

            case TokenType::KW_WHILE:
                new (&ast_line.line_branch) AstLineBranch();
                ast_line.ty = LineType::WHILE;

                end = tarr.search(IsSameCriteria(TokenType::COLON));
                if (end == -1)
                    return errs.add(tarr[i], "Missing ':' in while loop");
                if (end == i + 1)
                    return errs.add(tarr[i], "Empty expression in while loop condition");
                if (parse_expr(errs, { tarr, i + 1, end }, ast_line.line_branch.cond))
                    return errs.add(tarr[i], "Invalid syntax in while condition");

                for (end++; tarr[end]->ty == TokenType::INDENT; end++);
                if (tarr[end]->expect<TokenType::ENDL>(errs) ||
                    parse_lines<AstLine, parse_line>(errs, tarr, end, indent_level + 1, ast_line.line_branch.body))
                    return true;
                break;

            case TokenType::KW_FOR:
                new (&ast_line.line_for) AstLineFor();
                ast_line.ty = LineType::FOR;

                end = tarr.search(IsSameCriteria(TokenType::KW_IN), i);
                if (end == -1)
                    return errs.add(tarr[i], "Missing 'in' keyword in for loop");
                if (end == i + 1)
                    return errs.add(tarr[i], "Empty expression in for loop counter");
                if (parse_expr(errs, { tarr, i + 1, end }, ast_line.line_for.decl))
                    return errs.add(tarr[i], "Invalid syntax in for loop counter");
                i = end;
                
                end = tarr.search(IsSameCriteria(TokenType::COLON), i);
                if (end == -1)
                    return errs.add(tarr[i], "Missing ':' in for loop");
                if (end == i + 1)
                    return errs.add(tarr[i], "Empty expression in for loop iterator");
                if (parse_expr(errs, { tarr, i + 1, end }, ast_line.line_for.iter))
                    return errs.add(tarr[i], "Invalid syntax in for loop iterator");
                
                for (end++; tarr[end]->ty == TokenType::INDENT; end++);
                if (tarr[end]->expect<TokenType::ENDL>(errs) ||
                    parse_lines<AstLine, parse_line>(errs, tarr, end, indent_level + 1, ast_line.line_branch.body))
                    return true;
                break;
            }
            return false;
        }

        bool parse_arg_decl(Errors& errs, const TokenArray& tarr, AstArgDecl& ast_arg_decl)
        {
            // TODO: figure out how to handle fn and def references
            int i = tarr.size() - 1;
            for (; tarr[i]->is_whitespace(); i--);
            if (tarr[i]->expect<TokenType::IDN>(errs))
                return true;
            ast_arg_decl.var_name = tarr[i]->get<TokenType::IDN>().val;
            for (i--; tarr[i]->is_whitespace(); i--);
            if (tarr[i]->ty == TokenType::STAR)
                ast_arg_decl.is_packed = true;
            else
                i++;
            ast_arg_decl.type_expr = std::make_unique<AstExpr>();
            return parse_expr(errs, { tarr, 0, i }, *ast_arg_decl.type_expr);
        }

        bool parse_signature(Errors& errs, const TokenArray& tarr, AstStructSig& ast_struct_sig)
        {
            assert(tarr.size() > 0);

            // Setup and getting the name of the struct definition
            int i = 0;
            for (; tarr[i]->is_whitespace(); i++);
            if (tarr[i]->expect<TokenType::IDN>(errs))
                return true;
            ast_struct_sig.name = tarr[i]->get<TokenType::IDN>().val;
            for (i++; i < tarr.size() && tarr[i]->is_whitespace(); i++);
            if (i == tarr.size())
                return errs.add(tarr[tarr.size() - 1], "Unexpected end of signature after name");

            // Parsing out the cargs from the struct definition
            if (tarr[i]->ty == TokenType::ANGLE_O)
            {
                i = parse_args<AstArgDecl, parse_arg_decl, TokenType::ANGLE_O, TokenType::ANGLE_C>(errs, tarr, i, ast_struct_sig.cargs);
                if (i == -1)
                    return true;
                for (; i < tarr.size() && tarr[i]->is_whitespace(); i++);
                if (i == tarr.size())
                    return errs.add(tarr[tarr.size() - 1], "Unexpected end of signature after cargs");
            }
            return false;
        }

        bool parse_signature(Errors& errs, const TokenArray& tarr, AstFnSig& ast_fn_sig)
        {
            assert(tarr.size() > 0);

            // Setup and getting the name of the fn definition
            int i = 0;
            for (; tarr[i]->is_whitespace(); i++);
            if (tarr[i]->expect<TokenType::IDN>(errs))
                return true;
            ast_fn_sig.name = tarr[i]->get<TokenType::IDN>().val;
            for (i++; i < tarr.size() && tarr[i]->is_whitespace(); i++);
            if (i == tarr.size())
                return errs.add(tarr[tarr.size() - 1], "Unexpected end of signature after name");

            // Parsing out the cargs from the fn definition
            if (tarr[i]->ty == TokenType::ANGLE_O)
            {
                i = parse_args<AstArgDecl, parse_arg_decl, TokenType::ANGLE_O, TokenType::ANGLE_C>(errs, tarr, i, ast_fn_sig.cargs);
                if (i == -1)
                    return true;
                for (; i < tarr.size() && tarr[i]->is_whitespace(); i++);
                if (i == tarr.size())
                    return errs.add(tarr[tarr.size() - 1], "Unexpected end of signature after cargs");
            }

            // Parsing out the vargs from the fn definition
            if (tarr[i]->expect<TokenType::ROUND_O>(errs))
                return true;
            i = parse_args<AstArgDecl, parse_arg_decl, TokenType::ROUND_O, TokenType::ROUND_C>(errs, tarr, i, ast_fn_sig.vargs);
            if (i == -1)
                return true;
            for (; i < tarr.size() && tarr[i]->is_whitespace(); i++);
            if (i == tarr.size())
                return false;  // No return values

            // Parsing out the rets from the fn definition
            if (tarr[i]->expect<TokenType::ARROW>(errs))
                return true;
            for (i++; i < tarr.size() && tarr[i]->is_whitespace(); i++);
            if (i == tarr.size())
                return errs.add(tarr[tarr.size() - 1], "Expected an expression after '->'");
            // commas are valid operators in expression parsing resulting in a tuple
            // so I can reuse the expression parsing I already wrote to parse all the rets in the signature
            // then if a tuple is returned, I can just flatten it.  Otherwise, theres a single ret.
            AstExpr ret_expr;
            if (parse_expr(errs, { tarr, i }, ret_expr))
                return true;
            if (ret_expr.ty == ExprType::LIT_TUPLE)
                ast_fn_sig.rets = std::move(ret_expr.expr_agg.elems);
            else
                ast_fn_sig.rets.push_back(std::move(ret_expr));

            return false;
        }

        bool parse_signature(Errors& errs, const TokenArray& tarr, AstBlockSig& ast_block_sig)
        {
            assert(tarr.size() > 0);

            // Setup and getting the name of the block definition
            int i = 0;
            for (; tarr[i]->is_whitespace(); i++);
            if (tarr[i]->expect<TokenType::IDN>(errs))
                return true;
            ast_block_sig.name = tarr[i]->get<TokenType::IDN>().val;
            for (i++; i < tarr.size() && tarr[i]->is_whitespace(); i++);
            if (i == tarr.size())
                return errs.add(tarr[tarr.size() - 1], "Unexpected end of signature after name");

            // Parsing out the cargs from the block definition
            if (tarr[i]->ty == TokenType::ANGLE_O)
            {
                i = parse_args<AstArgDecl, parse_arg_decl, TokenType::ANGLE_O, TokenType::ANGLE_C>(errs, tarr, i, ast_block_sig.cargs);
                if (i == -1)
                    return true;
                for (; i < tarr.size() && tarr[i]->is_whitespace(); i++);
                if (i == tarr.size())
                    return errs.add(tarr[tarr.size() - 1], "Unexpected end of signature after cargs");
            }

            // Parsing out the vargs from the block definition
            if (tarr[i]->expect<TokenType::ROUND_O>(errs))
                return true;
            i = parse_args<AstArgDecl, parse_arg_decl, TokenType::ROUND_O, TokenType::ROUND_C>(errs, tarr, i, ast_block_sig.vargs);
            if (i == -1)
                return true;
            for (; i < tarr.size() && tarr[i]->is_whitespace(); i++);
            if (i == tarr.size())
                return false;  // No return values

            // Parsing out the first ret from the block definition
            if (tarr[i]->expect<TokenType::ARROW>(errs))
                return true;
            for (i++; i < tarr.size() && tarr[i]->is_whitespace(); i++);
            if (i == tarr.size())
                return errs.add(tarr[tarr.size() - 1], "Expected an indentifier after '->'");
            if (tarr[i]->expect<TokenType::IDN>(errs))
                return true;
            ast_block_sig.rets.push_back(tarr[i]->get<TokenType::IDN>().val);
            for (i++; i < tarr.size() && tarr[i]->is_whitespace(); i++);

            // Parsing out all the other rets
            while (i < tarr.size())
            {
                if (tarr[i]->expect<TokenType::COMMA>(errs))
                    return true;
                for (i++; i < tarr.size() && tarr[i]->is_whitespace(); i++);
                if (tarr[i]->expect<TokenType::IDN>(errs))
                    return true;
                ast_block_sig.rets.push_back(tarr[i]->get<TokenType::IDN>().val);
                for (i++; i < tarr.size() && tarr[i]->is_whitespace(); i++);
            }
            return false;
        }

        template<CodeBlockSig SIG>
        bool parse_code_block(Errors& errs, const TokenArray& tarr, AstCodeBlock<SIG>& ast_code_block)
        {
            assert(tarr.size() > 0);

            // Splitting the code block between the signature and the body
            int i = tarr.search(IsSameCriteria(TokenType::COLON));
            if (i == -1)
                return errs.add(tarr[0], "Missing ':' in signature");
            SIG signature{};
            if (parse_signature(errs, { tarr, 0, i }, signature))
                return true;

            // Finding the ':' that starts the body
            assert(!tarr[i]->expect<TokenType::COLON>(errs));
            for (i++; tarr[i]->ty == TokenType::INDENT; i++);  // one token past the last tab after the ':'
            if (tarr[i]->expect<TokenType::ENDL>(errs))
                return true;
            if (parse_lines<AstLine, parse_line>(errs, tarr, i, 1, ast_code_block.body))
                return true;
            ast_code_block.signature = signature;  // All the signature structs are movable
            return false;
        }

        template<> bool parse_code_block<AstStructSig>(Errors&, const TokenArray&, AstCodeBlock<AstStructSig>&);
        template<> bool parse_code_block<AstFnSig>    (Errors&, const TokenArray&, AstCodeBlock<AstFnSig>&    );
        template<> bool parse_code_block<AstBlockSig> (Errors&, const TokenArray&, AstCodeBlock<AstBlockSig>& );

        bool parse_import(Errors& errs, const TokenArray& tarr, AstImport& ast_import)
        {
            bool expect_idn = true;
            for (auto& tk : tarr)
            {
                if (tk.is_whitespace())
                    continue;
                if (expect_idn)
                {
                    if (tk.expect<TokenType::IDN>(errs))
                        return true;
                    ast_import.imp.push_back(tk.get<TokenType::IDN>().val);
                    expect_idn = false;
                }
                else
                {
                    if (tk.expect<TokenType::DOT>(errs))
                        return true;
                    expect_idn = true;
                }
            }
            return false;
        }

        bool parse_module(Errors& errs, const TokenArray& tarr, AstModule& ast_module)
        {
            ast_module.fname = tarr[0]->fname;

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
                        return errs.add(tarr[i], "Expected identifier after 'import'");
                    if (end < 0)
                        end = tarr.size();
                    ast_module.imports.push_back(AstImport());
                    if (!parse_import(errs, { tarr, i, end }, ast_module.imports.back()))
                        ast_module.imports.pop_back();
                    continue;
                case TokenType::KW_STRUCT:
                    i++;
                    end = tarr.search(LineEndCriteria(0), i);
                    if (end == i)
                        return errs.add(tarr[i], "Expected identifier after 'struct'");
                    if (end < 0)
                        end = tarr.size();
                    ast_module.structs.push_back(AstStruct());
                    if (parse_code_block(errs, { tarr, i, end }, ast_module.structs.back()))
                        ast_module.structs.pop_back();
                    continue;
                case TokenType::KW_FN:
                    i++;
                    end = tarr.search(LineEndCriteria(0), i);
                    if (end == i)
                        return errs.add(tarr[i], "Expected identifier after 'fn'");
                    if (end < 0)
                        end = tarr.size();
                    ast_module.funcs.push_back(AstFn());
                    if (parse_code_block(errs, { tarr, i, end }, ast_module.funcs.back()))
                        ast_module.funcs.pop_back();
                    continue;
                case TokenType::KW_DEF:
                    i++;
                    end = tarr.search(LineEndCriteria(0), i);
                    if (end == i)
                        return errs.add(tarr[i], "Expected identifier after 'def'");
                    if (end < 0)
                        end = tarr.size();
                    ast_module.defs.push_back(AstBlock());
                    if (parse_code_block(errs, { tarr, i, end }, ast_module.defs.back()))
                        ast_module.defs.pop_back();
                    continue;
                case TokenType::KW_INTR:
                    i++;
                    end = tarr.search(LineEndCriteria(0), i);
                    if (end == i)
                        return errs.add(tarr[i], "Expected identifier after 'intr'");
                    if (end < 0)
                        end = tarr.size();
                    ast_module.intrs.push_back(AstBlock());
                    if (parse_code_block(errs, { tarr, i, end }, ast_module.intrs.back()))
                        ast_module.intrs.pop_back();
                    continue;
                default:
                    return errs.add(tarr[i], "Invalid token");
                }
            }
            return false;
        }

        // From this point onward, the actual code ends and we enter c++ land

        AstExpr::AstExpr(AstExpr&& expr)
        {
            ty = expr.ty;
            expr.ty = ExprType::INVALID;

            switch (ty)
            {
            case ExprType::LIT_STRING:
            case ExprType::VAR:
                new (&expr_string) decltype(expr_string)(std::move(expr.expr_string));
                break;
            case ExprType::LIT_ARRAY:
            case ExprType::LIT_TUPLE:
                new (&expr_agg) decltype(expr_agg)(std::move(expr.expr_agg));
                break;
            case ExprType::UNARY_NOT:
            case ExprType::UNARY_POS:
            case ExprType::UNARY_NEG:
            case ExprType::UNARY_UNPACK:
                new (&expr_unary) decltype(expr_unary)(std::move(expr.expr_unary));
                break;
            case ExprType::BINARY_ADD:
            case ExprType::BINARY_SUB:
            case ExprType::BINARY_MUL:
            case ExprType::BINARY_DIV:
            case ExprType::BINARY_MOD:
            case ExprType::BINARY_IADD:
            case ExprType::BINARY_ISUB:
            case ExprType::BINARY_IMUL:
            case ExprType::BINARY_IDIV:
            case ExprType::BINARY_IMOD:
            case ExprType::BINARY_ASSIGN:
            case ExprType::BINARY_AND:
            case ExprType::BINARY_OR:
            case ExprType::BINARY_CMP_EQ:
            case ExprType::BINARY_CMP_NE:
            case ExprType::BINARY_CMP_GT:
            case ExprType::BINARY_CMP_LT:
            case ExprType::BINARY_CMP_GE:
            case ExprType::BINARY_CMP_LE:
            case ExprType::BINARY_IDX:
                new (&expr_binary) decltype(expr_binary)(std::move(expr.expr_binary));
                break;
            case ExprType::DOT:
            case ExprType::VAR_DECL:
                new (&expr_name) decltype(expr_name)(std::move(expr.expr_name));
                break;
            case ExprType::CARGS_CALL:
            case ExprType::VARGS_CALL:
                new (&expr_call) decltype(expr_call)(std::move(expr.expr_call));
                break;
            case ExprType::FN_DECL:
                new (&expr_fn_decl) decltype(expr_fn_decl)(std::move(expr.expr_fn_decl));
                break;
            case ExprType::DEF_DECL:
                new (&expr_def_decl) decltype(expr_def_decl)(std::move(expr.expr_def_decl));
                break;
            default:
                assert(false);
            }
        }

        AstExpr& AstExpr::operator=(AstExpr&& expr)
        {
            if (&expr == this)
                return *this;

            ty = expr.ty;
            expr.ty = ExprType::INVALID;

            switch (ty)
            {
            case ExprType::LIT_STRING:
            case ExprType::VAR:
                new (&expr_string) decltype(expr_string)(std::move(expr.expr_string));
                break;
            case ExprType::LIT_ARRAY:
            case ExprType::LIT_TUPLE:
                new (&expr_agg) decltype(expr_agg)(std::move(expr.expr_agg));
                break;
            case ExprType::UNARY_NOT:
            case ExprType::UNARY_POS:
            case ExprType::UNARY_NEG:
            case ExprType::UNARY_UNPACK:
                new (&expr_unary) decltype(expr_unary)(std::move(expr.expr_unary));
                break;
            case ExprType::BINARY_ADD:
            case ExprType::BINARY_SUB:
            case ExprType::BINARY_MUL:
            case ExprType::BINARY_DIV:
            case ExprType::BINARY_MOD:
            case ExprType::BINARY_IADD:
            case ExprType::BINARY_ISUB:
            case ExprType::BINARY_IMUL:
            case ExprType::BINARY_IDIV:
            case ExprType::BINARY_IMOD:
            case ExprType::BINARY_ASSIGN:
            case ExprType::BINARY_AND:
            case ExprType::BINARY_OR:
            case ExprType::BINARY_CMP_EQ:
            case ExprType::BINARY_CMP_NE:
            case ExprType::BINARY_CMP_GT:
            case ExprType::BINARY_CMP_LT:
            case ExprType::BINARY_CMP_GE:
            case ExprType::BINARY_CMP_LE:
            case ExprType::BINARY_IDX:
                new (&expr_binary) decltype(expr_binary)(std::move(expr.expr_binary));
                break;
            case ExprType::DOT:
            case ExprType::VAR_DECL:
                new (&expr_name) decltype(expr_name)(std::move(expr.expr_name));
                break;
            case ExprType::CARGS_CALL:
            case ExprType::VARGS_CALL:
                new (&expr_call) decltype(expr_call)(std::move(expr.expr_call));
                break;
            case ExprType::FN_DECL:
                new (&expr_fn_decl) decltype(expr_fn_decl)(std::move(expr.expr_fn_decl));
                break;
            case ExprType::DEF_DECL:
                new (&expr_def_decl) decltype(expr_def_decl)(std::move(expr.expr_def_decl));
                break;
            default:
                assert(false);
            }

            return *this;
        }

        AstExpr::~AstExpr()
        {
            switch (ty)
            {
            case ExprType::LIT_STRING:
            case ExprType::VAR:
                expr_string.~basic_string();
                break;
            case ExprType::LIT_ARRAY:
            case ExprType::LIT_TUPLE:
                expr_agg.~AstExprAggLit();
                break;
            case ExprType::UNARY_NOT:
            case ExprType::UNARY_POS:
            case ExprType::UNARY_NEG:
            case ExprType::UNARY_UNPACK:
                expr_unary.~AstExprUnaryOp();
                break;
            case ExprType::BINARY_ADD:
            case ExprType::BINARY_SUB:
            case ExprType::BINARY_MUL:
            case ExprType::BINARY_DIV:
            case ExprType::BINARY_MOD:
            case ExprType::BINARY_IADD:
            case ExprType::BINARY_ISUB:
            case ExprType::BINARY_IMUL:
            case ExprType::BINARY_IDIV:
            case ExprType::BINARY_IMOD:
            case ExprType::BINARY_ASSIGN:
            case ExprType::BINARY_AND:
            case ExprType::BINARY_OR:
            case ExprType::BINARY_CMP_EQ:
            case ExprType::BINARY_CMP_NE:
            case ExprType::BINARY_CMP_GT:
            case ExprType::BINARY_CMP_LT:
            case ExprType::BINARY_CMP_GE:
            case ExprType::BINARY_CMP_LE:
            case ExprType::BINARY_IDX:
                expr_binary.~AstExprBinaryOp();
                break;
            case ExprType::DOT:
            case ExprType::VAR_DECL:
                expr_name.~AstExprName();
                break;
            case ExprType::CARGS_CALL:
            case ExprType::VARGS_CALL:
                expr_call.~AstExprCall();
                break;
            case ExprType::FN_DECL:
                expr_fn_decl.~AstFnSig();
                break;
            case ExprType::DEF_DECL:
                expr_def_decl.~AstBlockSig();
                break;
            default:
                assert(false);
            }
        }

        AstLine::AstLine(AstLine&& line)
        {
            ty = line.ty;
            line.ty = LineType::INVALID;

            switch (ty)
            {
            case LineType::EXPORT:
                new (&line_export) decltype(line_export)(std::move(line.line_export));
                break;
            case LineType::RAISE:
            case LineType::PRINT:
            case LineType::RETURN:
                new (&line_func) decltype(line_func)(std::move(line.line_func));
                break;
            case LineType::IF:
            case LineType::ELIF:
            case LineType::WHILE:
                new (&line_branch) decltype(line_branch)(std::move(line.line_branch));
                break;
            case LineType::ELSE:
                new (&line_else) decltype(line_else)(std::move(line.line_else));
                break;
            case LineType::FOR:
                new (&line_for) decltype(line_for)(std::move(line.line_for));
                break;
            case LineType::EXPR:
                new (&line_expr) decltype(line_expr)(std::move(line.line_expr));
                break;
            default:
                assert(false);
            }
        }

        AstLine& AstLine::operator=(AstLine&& line)
        {
            if (&line == this)
                return *this;

            ty = line.ty;
            line.ty = LineType::INVALID;

            switch (ty)
            {
            case LineType::EXPORT:
                new (&line_export) decltype(line_export)(std::move(line.line_export));
                break;
            case LineType::RAISE:
            case LineType::PRINT:
            case LineType::RETURN:
                new (&line_func) decltype(line_func)(std::move(line.line_func));
                break;
            case LineType::IF:
            case LineType::ELIF:
            case LineType::WHILE:
                new (&line_branch) decltype(line_branch)(std::move(line.line_branch));
                break;
            case LineType::ELSE:
                new (&line_else) decltype(line_else)(std::move(line.line_else));
                break;
            case LineType::FOR:
                new (&line_for) decltype(line_for)(std::move(line.line_for));
                break;
            case LineType::EXPR:
                new (&line_expr) decltype(line_expr)(std::move(line.line_expr));
                break;
            default:
                assert(false);
            }

            return *this;
        }

        AstLine::~AstLine()
        {
            switch (ty)
            {
            case LineType::EXPORT:
                line_export.~AstLineExport();
                break;
            case LineType::RAISE:
            case LineType::PRINT:
            case LineType::RETURN:
                line_func.~AstLineUnaryFunc();
                break;
            case LineType::IF:
            case LineType::ELIF:
            case LineType::WHILE:
                line_branch.~AstLineBranch();
                break;
            case LineType::ELSE:
                line_else.~AstLineElse();
                break;
            case LineType::FOR:
                line_for.~AstLineFor();
                break;
            case LineType::EXPR:
                line_expr.~AstLineExpr();
                break;
            default:
                assert(false);
            }
        }
    }
}
