#include <ned/lang/ast.h>
#include <ned/lang/obj.h>

#include <functional>
#include <cassert>

using namespace nn::lang;

// Parses comma deliminated token sequences within a matching set of brackets
// Returns the index in tarr immediately after the CLOSE token was found
// On failure, this function returns -1
template<typename T, bool(*parse_fn)(ParsingErrors&, const TokenArray&, T&), TokenType OPEN, TokenType CLOSE>
inline int parse_args(ParsingErrors& errs, const TokenArray& tarr, int i, std::vector<T>& args)
{
    assert(tarr[0]->ty == OPEN);
    
    // Advancing one token past the opening bracket
    i++;
    int end = tarr.search(CargEndCriteria(), i);
    if (tarr[end]->ty == CLOSE)
    {
        // either a single carg, or its empty
        for (int j = i; j < end; j++)
            if (!tarr[j]->is_whitespace())
            {
                // single carg, parse it and move on
                args.push_back(T());
                if (parse_fn(errs, { tarr, i, end }, args.back()))
                {
                    args.clear();
                    return -1;
                }
                break;
            }
        i = end;
        continue;
    }

    // General case for parsing multiple cargs
    args.push_back(T());
    if (parse_fn({ tarr, i, end }, args.back()))
    {
        args.clear();
        return -1;
    }
    do
    {
        end = tarr.search(CargEndCriteria(), i);
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
template<typename T, bool(*parse_fn)(ParsingErrors&, const TokenArray&, T&, int)>
inline bool parse_lines(ParsingErrors& errs, const TokenArray& tarr, int i, int indent_level, std::vector<T>& lines)
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

// Parses leaf expressions
bool parse_expr_leaf(ParsingErrors& errs, const TokenArray& tarr, AstExpr& ast_expr)
{

}

// *, /, %
bool parse_expr_muldivmod(ParsingErrors& errs, const TokenArray& tarr, AstExpr& ast_expr);

// Addition / subtraction
bool parse_expr_addsub(ParsingErrors& errs, const TokenArray& tarr, AstExpr& ast_expr);

// Comparision operators
bool parse_expr_compare(ParsingErrors& errs, const TokenArray& tarr, AstExpr& ast_expr);

// Logical operators: and, or
bool parse_expr_logic(ParsingErrors& errs, const TokenArray& tarr, AstExpr& ast_expr);

// Assignment operators: =, +=, %=
bool parse_expr_assign(ParsingErrors& errs, const TokenArray& tarr, AstExpr& ast_expr);

enum class OpAssoc
{
    LTOR,
    RTOL
};

const std::vector<std::tuple<OpAssoc, std::vector<TokenType>>> prec_ops = {
    { OpAssoc::LTOR, { TokenType::COMMA }},
    { OpAssoc::RTOL, { TokenType::ASSIGN, TokenType::IADD, TokenType::ISUB, TokenType::IMUL, TokenType::IDIV, TokenType::IMOD }},
    { OpAssoc::LTOR, { TokenType::KW_AND, TokenType::KW_OR }},
    { OpAssoc::LTOR, { TokenType::CMP_EQ, TokenType::CMP_NE, TokenType::CMP_GT, TokenType::CMP_LT, TokenType::CMP_GE, TokenType::CMP_LE }},
    { OpAssoc::LTOR, { TokenType::ADD, TokenType::SUB }},
    { OpAssoc::LTOR, { TokenType::STAR, TokenType::DIV, TokenType::MOD }}
};

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
}

template<int prec>
bool parse_expr_prec(ParsingErrors& errs, const TokenArray& tarr, AstExpr& ast_expr);

template<int prec>
bool parse_expr_prec(ParsingErrors& errs, const TokenArray& tarr, AstExpr& ast_expr)
{
    assert(tarr.size() > 0);
    constexpr auto& [op_assoc, tys] = prec_ops[prec];
    int pos;
    if (op_assoc == OpAssoc::LTOR)
        pos = tarr.search(IsInCriteria(tys), 1, -1);
    else
        pos = tarr.rsearch(IsInCriteria(tys));
    if (pos == -1)
        return parse_expr_prec<prec + 1>(errs, tarr, ast_expr);
    if (pos < 1 || tarr.size() - pos < 2)
        return errs.add(tarr[pos], "Dangling operator");
    AstExpr* pleft;
    AstExpr* pright;
    pleft = new AstExpr();
    if (parse_expr_prec<prec + 1>(errs, { tarr, 0, pos }, *pleft))
        goto LEFT_ERR;
    pright = new AstExpr();
    if (parse_expr_prec<prec + 1>(errs, { tarr, pos + 1, tarr.size() }, *pright))
        goto RIGHT_ERR;
    ast_expr.expr_binary.left = pleft;
    ast_expr.expr_binary.right = pright;
    ast_expr.ty = get_expr_type<prec>(tarr[pos]->ty);
    return false;

RIGHT_ERR:
    delete pright;
LEFT_ERR:
    delete pleft;
    return true;
}

template<>
bool parse_expr_prec<7>(ParsingErrors& errs, const TokenArray& tarr, AstExpr& ast_expr)
{
    assert(tarr.size() > 0);
    int i = 0;
    for (; i < tarr.size() && tarr[i]->is_whitespace(); i++);
    if (i == tarr.size())
        return errs.add(tarr[0], "Invalid expression");
    
    AstExpr* pexpr;
    switch (tarr[i]->ty)
    {
    case TokenType::ADD:
        if (tarr.size() - i < 2)
            return errs.add(tarr[i], "Dangling operator");
        pexpr = new AstExpr();
        if (parse_expr_prec<8>(errs, { tarr, i + 1, tarr.size() }, *pexpr))
        {
            delete pexpr;
            return true;
        }
        ast_expr.expr_unary.expr = pexpr;
        ast_expr.ty = ExprType::UNARY_POS;
        return false;
    case TokenType::SUB:
        if (tarr.size() - i < 2)
            return errs.add(tarr[i], "Dangling operator");
        pexpr = new AstExpr();
        if (parse_expr_prec<8>(errs, { tarr, i + 1, tarr.size() }, *pexpr))
        {
            delete pexpr;
            return true;
        }
        ast_expr.expr_unary.expr = pexpr;
        ast_expr.ty = ExprType::UNARY_NEG;
        return false;
    case TokenType::KW_NOT:
        if (tarr.size() - i < 2)
            return errs.add(tarr[i], "Dangling operator");
        pexpr = new AstExpr();
        if (parse_expr_prec<8>(errs, { tarr, i + 1, tarr.size() }, *pexpr))
        {
            delete pexpr;
            return true;
        }
        ast_expr.expr_unary.expr = pexpr;
        ast_expr.ty = ExprType::UNARY_NOT;
        return false;
    }

    int end = tarr.size() - 1;
    for (; tarr[end]->is_whitespace(); end--);
    return parse_expr_prec<8>(errs, { tarr, i, end + 1 }, ast_expr);
}

namespace nn
{
    namespace lang
    {
        bool parse_expr(ParsingErrors& errs, const TokenArray& tarr, AstExpr& ast_expr)
        {
            
            
            return true;
        }

        bool parse_line(ParsingErrors& errs, const TokenArray& tarr, AstLine& ast_line, int indent_level)
        {
            assert(tarr.size() > 0);
            int i = 0;
            for (; i < tarr.size() && tarr[i]->is_whitespace(); i++);
            if (i == tarr.size())
                return errs.add(tarr[0], "Invalid line");
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
                ast_line.line_export.var_name = tarr[i]->get<TokenType::IDN>().val;
                ast_line.ty = LineType::EXPORT;
                break;
            case TokenType::KW_RAISE:
                if (parse_expr(errs, { tarr, i + 1 }, ast_line.line_func.expr))
                    return errs.add(tarr[i], "Invalid syntax in raise statement");
                ast_line.ty = LineType::RAISE;
                break;
            case TokenType::KW_RETURN:
                if (parse_expr(errs, { tarr, i + 1 }, ast_line.line_func.expr))
                    return errs.add(tarr[i], "Invalid syntax in return statement");
                ast_line.ty = LineType::RETURN;
                break;
            case TokenType::KW_PRINT:
                if (parse_expr(errs, { tarr, i + 1 }, ast_line.line_func.expr))
                    return errs.add(tarr[i], "Invalid syntax in print statement");
                ast_line.ty = LineType::PRINT;
                break;
            case TokenType::KW_IF:
                end = tarr.search(IsSameCriteria(TokenType::COLON));
                if (parse_expr(errs, { tarr, i + 1, end }, ast_line.line_branch.cond))
                    return errs.add(tarr[i], "Invalid syntax in if condition");
                for (; tarr[end]->ty == TokenType::INDENT; end++);
                if (tarr[end]->expect<TokenType::ENDL>(errs) ||
                    parse_lines<AstLine, parse_line>(errs, tarr, end, indent_level + 1, ast_line.line_branch.body))
                {
                    ast_line.line_branch.cond.~AstExpr();
                    return true;
                }
                ast_line.ty = LineType::IF;
                break;
            case TokenType::KW_ELIF:
                end = tarr.search(IsSameCriteria(TokenType::COLON));
                if (parse_expr(errs, { tarr, i + 1, end }, ast_line.line_branch.cond))
                    return errs.add(tarr[i], "Invalid syntax in elif condition");
                for (; tarr[end]->ty == TokenType::INDENT; end++);
                if (tarr[end]->expect<TokenType::ENDL>(errs) ||
                    parse_lines<AstLine, parse_line>(errs, tarr, end, indent_level + 1, ast_line.line_branch.body))
                {
                    ast_line.line_branch.cond.~AstExpr();
                    return true;
                }
                ast_line.ty = LineType::ELIF;
                break;
            case TokenType::KW_ELSE:
                end = tarr.search(IsSameCriteria(TokenType::COLON));
                for (; tarr[end]->ty == TokenType::INDENT; end++);
                if (tarr[end]->expect<TokenType::ENDL>(errs))
                    return true;
                if (parse_lines<AstLine, parse_line>(errs, tarr, end, indent_level + 1, ast_line.line_else.body))
                    return errs.add(tarr[i], "Invalid syntax in else block");
                ast_line.ty = LineType::ELSE;
                break;
            case TokenType::KW_WHILE:
                end = tarr.search(IsSameCriteria(TokenType::COLON));
                if (parse_expr(errs, { tarr, i + 1, end }, ast_line.line_branch.cond))
                    return errs.add(tarr[i], "Invalid syntax in while condition");
                for (; tarr[end]->ty == TokenType::INDENT; end++);
                if (tarr[end]->expect<TokenType::ENDL>(errs) ||
                    parse_lines<AstLine, parse_line>(errs, tarr, end, indent_level + 1, ast_line.line_branch.body))
                {
                    ast_line.line_branch.cond.~AstExpr();
                    return true;
                }
                ast_line.ty = LineType::WHILE;
                break;
            case TokenType::KW_FOR:
                end = tarr.search(IsSameCriteria(TokenType::KW_IN), i);
                if (end == -1)
                    return errs.add(tarr[i], "Missing 'in' keyword in for loop");
                {
                    if (parse_expr(errs, { tarr, i + 1, end }, ast_line.line_for.decl))
                        return errs.add(tarr[0], "Invalid syntax in for loop counter");
                    i = end + 1;
                    end = tarr.search(IsSameCriteria(TokenType::COLON), i);
                    AstExpr expr{};
                    parse_expr(errs, { tarr, i, end }, expr);
                    // TODO: continue this thing
                }
                ast_line.ty = LineType::FOR;
                break;
            }
            return false;
        }

        bool parse_arg_decl(ParsingErrors& errs, const TokenArray& tarr, AstArgDecl& ast_arg_decl)
        {
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
            if (parse_expr(errs, { tarr, 0, i }, ast_arg_decl.type_expr))
            {
                ast_arg_decl.var_name.~basic_string();
                return true;
            }
            return false;
        }

        bool parse_callable(ParsingErrors& errs, const TokenArray& tarr, AstCallable& ast_callable)
        {
            // Setup and getting the name of the callable definition
            int i = 0;
            for (; tarr[i]->is_whitespace(); i++);
            if (tarr[i]->expect<TokenType::IDN>(errs))
                return true;
            ast_callable.name = tarr[i]->get<TokenType::IDN>().val;

            // Parsing out the cargs from the callable definition
            for (; tarr[i]->is_whitespace(); i++);
            if (tarr[i]->ty == TokenType::ANGLE_O)
            {
                i = parse_args<AstArgDecl, parse_arg_decl, TokenType::ANGLE_O, TokenType::ANGLE_C>(errs, tarr, i, ast_callable.cargs);
                if (i == -1)
                    return true;
                for (; tarr[i]->is_whitespace(); i++);
            }

            // Parsing out the vargs from the callable definition
            if (tarr[i]->ty == TokenType::ROUND_O)
            {
                i = parse_args<AstArgDecl, parse_arg_decl, TokenType::ROUND_O, TokenType::ROUND_C>(errs, tarr, i, ast_callable.vargs);
                if (i == -1)
                    goto VARG_ERR;
                for (; tarr[i]->is_whitespace(); i++);
            }

            // Finding the ':' that starts the body
            if (tarr[i]->expect<TokenType::COLON>(errs))
                goto BODY_ERR;
            for (; tarr[i]->ty == TokenType::INDENT; i++);  // one token past the first endl after the ':'
            if (tarr[i]->expect<TokenType::ENDL>(errs))
                goto BODY_ERR;
            if (parse_lines<AstLine, parse_line>(errs, tarr, i, 1, ast_callable.lines))
                goto BODY_ERR;
            return false;

        BODY_ERR:
            ast_callable.vargs.clear();
        VARG_ERR:
            ast_callable.cargs.clear();
            return true;
        }

        bool parse_struct(ParsingErrors& errs, const TokenArray& tarr, AstStruct& ast_struct)
        {
            // Setup and getting the name of the struct definition
            int i = 0;
            for (; tarr[i]->is_whitespace(); i++);
            ast_struct.name = tarr[i]->get<TokenType::IDN>().val;

            // Parsing out the cargs from the struct if it has any
            for (; tarr[i]->is_whitespace(); i++);
            if (tarr[i]->ty == TokenType::ANGLE_O)
            {
                i = parse_args<AstArgDecl, parse_arg_decl, TokenType::ANGLE_O, TokenType::ANGLE_C>(errs, tarr, i, ast_struct.cargs);
                for (; tarr[i]->is_whitespace(); i++);
            }

            // Finding the ':' that starts the struct body
            tarr[i]->get<TokenType::COLON>();
            for (; tarr[i]->ty == TokenType::INDENT; i++);  // one token past the first endl after the ':'
            tarr[i]->get<TokenType::ENDL>();
            parse_lines<AstLine, parse_line>(errs, tarr, i, 1, ast_struct.decls);
        }

        bool parse_import(ParsingErrors& errs, const TokenArray& tarr, AstImport& ast_import)
        {
            bool expect_idn = true;
            for (auto& tk : tarr)
            {
                if (tk.is_whitespace())
                    continue;
                if (expect_idn)
                {
                    if (tk.expect<TokenType::IDN>(errs)) break;
                    ast_import.imp.push_back(tk.get<TokenType::IDN>().val);
                    expect_idn = false;
                }
                else
                {
                    if (tk.expect<TokenType::DOT>(errs)) break;
                    expect_idn = true;
                }
            }
        }

        bool parse_module(ParsingErrors& errs, const TokenArray& tarr, AstModule& ast_module)
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
                    parse_import(errs, { tarr, i, end }, ast_module.imports.back());
                    continue;
                case TokenType::KW_STRUCT:
                    i++;
                    end = tarr.search(LineEndCriteria(0), i);
                    if (end == i)
                        return errs.add(tarr[i], "Expected identifier after 'struct'");
                    if (end < 0)
                        end = tarr.size();
                    ast_module.structs.push_back(AstStruct());
                    parse_struct(errs, { tarr, i, end }, ast_module.structs.back());
                    continue;
                case TokenType::KW_FN:
                    i++;
                    end = tarr.search(LineEndCriteria(0), i);
                    if (end == i)
                        return errs.add(tarr[i], "Expected identifier after 'fn'");
                    if (end < 0)
                        end = tarr.size();
                    ast_module.funcs.push_back(AstCallable());
                    parse_callable(errs, { tarr, i, end }, ast_module.funcs.back());
                    continue;
                case TokenType::KW_DEF:
                    i++;
                    end = tarr.search(LineEndCriteria(0), i);
                    if (end == i)
                        return errs.add(tarr[i], "Expected identifier after 'def'");
                    if (end < 0)
                        end = tarr.size();
                    ast_module.defs.push_back(AstCallable());
                    parse_callable(errs, { tarr, i, end }, ast_module.defs.back());
                    continue;
                case TokenType::KW_INTR:
                    i++;
                    end = tarr.search(LineEndCriteria(0), i);
                    if (end == i)
                        return errs.add(tarr[i], "Expected identifier after 'intr'");
                    if (end < 0)
                        end = tarr.size();
                    ast_module.intrs.push_back(AstCallable());
                    parse_callable(errs, { tarr, i, end }, ast_module.intrs.back());
                    continue;
                default:
                    return errs.add(tarr[i], "Invalid token");
                }
            }
        }

        AstExprUnaryOp::~AstExprUnaryOp()
        {
            delete expr;
        }

        AstExprBinaryOp::~AstExprBinaryOp()
        {
            delete left;
            delete right;
        }

        AstExprDot::~AstExprDot()
        {
            delete expr;
        }

        AstExprCall::~AstExprCall()
        {
            delete callee;
        }

        AstExprDecl::~AstExprDecl()
        {
            delete type;
        }

        AstExpr::~AstExpr()
        {
            switch (ty)
            {
            case ExprType::INVALID:
            case ExprType::LIT_BOOL:
            case ExprType::LIT_INT:
            case ExprType::LIT_FLOAT:
                break;  // Empty or POD
            case ExprType::LIT_STRING:
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
                expr_binary.~AstExprBinaryOp();
                break;
            case ExprType::DOT:
                expr_dot.~AstExprDot();
            case ExprType::CARGS_CALL:
            case ExprType::VARGS_CALL:
                expr_call.~AstExprCall();
                break;
            case ExprType::DECL:
                expr_decl.~AstExprDecl();
                break;
            }
        }

        AstLine::~AstLine()
        {
            switch (ty)
            {
            case LineType::INVALID:
            case LineType::BREAK:
            case LineType::CONTINUE:
                break;  // Empty
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
            }
        }
    }
}
