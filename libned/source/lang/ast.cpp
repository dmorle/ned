#include <ned/lang/ast.h>
#include <ned/lang/obj.h>

#include <functional>
#include <cassert>

using namespace nn::lang;

// Parses comma deliminated token sequences within a matching set of brackets
// Returns the index in tarr immediately after the CLOSE token was found
template<typename T, std::function<T(const TokenArray&)> parse_fn, TokenType OPEN, TokenType CLOSE>
inline int parse_args(int i, std::vector<T>& args, const TokenArray& tarr)
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
                args.push_back(parse_fn({ tarr, i, end }));
                break;
            }
        i = end;
        continue;
    }

    // General case for parsing multiple cargs
    args.push_back(parse_fn({ tarr, i, end }));
    do
    {
        end = tarr.search(CargEndCriteria(), i);
        args.push_back(parse_fn({ tarr, i, end }));
        i = end + 1;
    } while (tarr[end]->ty == TokenType::COMMA);

    return i;
}

// Parses lines from a token sequence
template<typename T, T(*parse_fn)(const TokenArray&)>
inline void parse_lines(int i, int indent_level, std::vector<T>& lines, const TokenArray& tarr)
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
                lines.push_back(parse_fn({ tarr, i, end }));
                break;
            }
        i = end + 1;
    } while (end < tarr.size());
}

// Implementation of parse_expr() and parse_expr_alloc()
void parse_expr_impl(const TokenArray& tarr, AstExpr& ast_expr)
{

}

// Implementation of parse_line() and parse_line_alloc()
void parse_line_impl(const TokenArray& tarr, AstLine& ast_line)
{
    int i = 0;
    for (i++; tarr[i]->is_whitespace(); i++);
    int end;
    switch (tarr[i]->ty)
    {
    case TokenType::KW_BREAK:
        for (const auto& tk : TokenArray{ tarr, 1 })
            if (!tk.is_whitespace())
                throw SyntaxError(tarr, "Unexpected token in 'break' statement");
        ast_line.ty = LineType::BREAK;
        return ast_line;
    case TokenType::KW_CONTINUE:
        for (const auto& tk : TokenArray{ tarr, 1 })
            if (!tk.is_whitespace())
                throw SyntaxError(tarr, "Unexpected token in 'continue' statement");
        ast_line.ty = LineType::CONTINUE;
        return ast_line;
    case TokenType::KW_EXPORT:
        for (i++; tarr[i]->is_whitespace(); i++);
        ast_line.line_export.var_name = tarr[i]->get<TokenType::IDN>().val;
        ast_line.ty = LineType::EXPORT;
        return ast_line;
    case TokenType::KW_RAISE:
        ast_line.line_func.expr = parse_expr({ tarr, i + 1 });
        ast_line.ty = LineType::RAISE;
        break;
    case TokenType::KW_RETURN:
        ast_line.line_func.expr = parse_expr({ tarr, i + 1 });
        ast_line.ty = LineType::RETURN;
        break;
    case TokenType::KW_PRINT:
        for (i++; tarr[i]->is_whitespace(); i++);
        tarr[i]->get<TokenType::ROUND_O>();
        end = = tarr.size();
        for (end--; tarr[i]->is_whitespace(); end--);
        tarr[end]->get<TokenType::ROUND_C>();
        ast_line.line_func.expr = parse_expr({ tarr, i + 1, end });
        ast_line.ty = LineType::PRINT;
        break;
    case TokenType::KW_IF:
        end = tarr->search(IsSameCriteria(TokenType::COLON));
        {
            AstExpr cond = parse_expr({ tarr, i + 1, end });
            for (; tarr[end]->ty == TokenType::INDENT; end++);
            tarr[end]->get<TokenType::ENDL>();
            ast_line.line_branch.body = parse_lines<AstLine, parse_line>(end, 1, ast_callable.lines, tarr);
            ast_line.line_branch.cond = cond;
        }
        ast_line.ty = LineType::IF;
        break;
    case TokenType::KW_ELIF:
        end = tarr->search(IsSameCriteria(TokenType::COLON));
        {
            AstExpr cond = parse_expr({ tarr, i + 1, end });
            for (; tarr[end]->ty == TokenType::INDENT; end++);
            tarr[end]->get<TokenType::ENDL>();
            parse_lines<AstLine, parse_line>(end, 1, ast_line.line_branch.body, tarr);
            ast_line.line_branch.cond = cond;
        }
        ast_line.ty = LineType::ELIF;
        break;
    case TokenType::KW_ELSE:
        end = tarr->search(IsSameCriteria(TokenType::COLON));
        for (; tarr[end]->ty == TokenType::INDENT; end++);
        tarr[end]->get<TokenType::ENDL>();
        parse_lines<AstLine, parse_line>(end, 1, ast_line.line_else.body, tarr);
        ast_line.ty = LineType::ELSE;
        break;
    case TokenType::KW_WHILE:
        end = tarr->search(IsSameCriteria(TokenType::COLON));
        {
            AstExpr cond = parse_expr({ tarr, i + 1, end });
            for (; tarr[end]->ty == TokenType::INDENT; end++);
            tarr[end]->get<TokenType::ENDL>();
            ast_line.line_branch.body = parse_lines<AstLine, parse_line>(end, 1, ast_callable.lines, tarr);
            ast_line.line_branch.cond = cond;
        }
        ast_line.ty = LineType::WHILE;
        break;
    case TokenType::KW_FOR:
        end = tarr->search(IsSameCriteria(TokenType::KW_IN), i);
        if (end == -1)
            throw SyntaxError(tarr[i], "Missing 'in' keyword in for loop");
        {
            AstLineDecl decl = parse_line_decl({ tarr, i + 1, end });
            i = end + 1;
            end = tarr->search(IsSameCriteria(TokenType::COLON), i);
            AstExpr expr = parse_expr({ tarr, i, end });
            // TODO: continue this thing
        }
        ast_line.ty = LineType::FOR;
        break;
    }
}

namespace nn
{
    namespace lang
    {
        AstExpr* parse_expr_alloc(const TokenArray& tarr)
        {
            AstExpr* ast_expr = new AstExpr();
            parse_expr_impl(tarr, *ast_expr);
            return ast_expr;
        }

        AstExpr parse_expr(const TokenArray& tarr)
        {
            AstExpr ast_expr;
            parse_expr_impl(tarr, ast_expr);
            return ast_expr;
        }

        AstLine* parse_line_alloc(const TokenArray& tarr)
        {
            AstLine* ast_line = new AstLine();
            parse_line_impl(tarr, *ast_line);
            return ast_line;
        }

        AstLine parse_line(const TokenArray& tarr)
        {
            AstLine ast_line;
            parse_line_impl(tarr, ast_line);
            return ast_line;
        }

        AstArgDecl parse_arg_decl(const TokenArray& tarr)
        {
            AstArgDecl ast_arg_decl;

            int i = tarr.size() - 1;
            for (; tarr[i]->is_whitespace(); i--);
            ast_arg_decl.var_name = tarr[i]->get<TokenType::IDN>().val;
            for (i--; tarr[i]->is_whitespace(); i--);
            if (tarr[i]->ty == TokenType::STAR)
                ast_arg_decl.is_packed = true;
            else
                i++;
            ast_arg_decl.type_expr = parse_expr({ tarr, 0, i });
        }

        AstCallable parse_callable(const TokenArray& tarr)
        {
            AstCallable ast_callable;

            // Setup and getting the name of the callable definition
            int i = 0;
            for (; tarr[i]->is_whitespace(); i++);
            ast_callable.name = tarr[i]->get<TokenType::IDN>().val;

            // Parsing out the cargs from the callable definition
            for (; tarr[i]->is_whitespace(); i++);
            if (tarr[i]->ty == TokenType::ANGLE_O)
            {
                i = parse_args<AstArgDecl, parse_arg_decl, TokenType::ANGLE_O, TokenType::ANGLE_C>(i, ast_callable.cargs, tarr);
                for (; tarr[i]->is_whitespace(); i++);
            }

            // Parsing out the vargs from the callable definition
            if (tarr[i]->ty == TokenType::ROUND_O)
            {
                i = parse_args<AstArgDecl, parse_arg_decl, TokenType::ROUND_O, TokenType::ROUND_C>(i, ast_callable.vargs, tarr);
                for (; tarr[i]->is_whitespace(); i++);
            }

            // Finding the ':' that starts the struct body
            tarr[i]->get<TokenType::COLON>();
            for (; tarr[i]->ty == TokenType::INDENT; i++);  // one token past the first endl after the ':'
            tarr[i]->get<TokenType::ENDL>();
            parse_lines<AstLine, parse_line>(i, 1, ast_callable.lines, tarr);

            return ast_callable;
        }

        AstStruct parse_struct(const TokenArray& tarr)
        {
            AstStruct ast_struct;

            // Setup and getting the name of the struct definition
            int i = 0;
            for (; tarr[i]->is_whitespace(); i++);
            ast_struct.name = tarr[i]->get<TokenType::IDN>().val;

            // Parsing out the cargs from the struct if it has any
            for (; tarr[i]->is_whitespace(); i++);
            if (tarr[i]->ty == TokenType::ANGLE_O)
            {
                i = parse_args<AstArgDecl, parse_arg_decl, TokenType::ANGLE_O, TokenType::ANGLE_C>(i, ast_struct.cargs, tarr);
                for (; tarr[i]->is_whitespace(); i++);
            }

            // Finding the ':' that starts the struct body
            tarr[i]->get<TokenType::COLON>();
            for (; tarr[i]->ty == TokenType::INDENT; i++);  // one token past the first endl after the ':'
            tarr[i]->get<TokenType::ENDL>();
            parse_lines<AstLineDecl, parse_line_decl>(i, 1, ast_struct.decls, tarr);

            return ast_struct;
        }

        AstImport parse_import(const TokenArray& tarr)
        {
            AstImport ast_import;

            bool expect_idn = true;
            for (auto& tk : tarr)
            {
                if (tk.is_whitespace())
                    continue;
                if (expect_idn)
                {
                    ast_import.imp.push_back(tk.get<TokenType::IDN>().val);
                    expect_idn = false;
                }
                else
                {
                    tk.get<TokenType::DOT>();
                    expect_idn = true;
                }
            }
            return ast_import;
        }

        AstModule parse_module(const TokenArray& tarr)
        {
            AstModule ast_module;
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
                        throw SyntaxError(tarr[i], "Expected identifier after 'import'");
                    if (end < 0)
                        end = tarr.size();
                    ast_module.imports.push_back(parse_import({ tarr, i, end }));
                    continue;
                case TokenType::KW_STRUCT:
                    i++;
                    end = tarr.search(LineEndCriteria(0), i);
                    if (end == i)
                        throw SyntaxError(tarr[i], "Expected identifier after 'struct'");
                    if (end < 0)
                        end = tarr.size();
                    ast_module.structs.push_back(parse_struct({ tarr, i, end }));
                    continue;
                case TokenType::KW_FN:
                    i++;
                    end = tarr.search(LineEndCriteria(0), i);
                    if (end == i)
                        throw SyntaxError(tarr[i], "Expected identifier after 'fn'");
                    if (end < 0)
                        end = tarr.size();
                    ast_module.funcs.push_back(parse_callable({ tarr, i, end }));
                    continue;
                case TokenType::KW_DEF:
                    i++;
                    end = tarr.search(LineEndCriteria(0), i);
                    if (end == i)
                        throw SyntaxError(tarr[i], "Expected identifier after 'def'");
                    if (end < 0)
                        end = tarr.size();
                    ast_module.defs.push_back(parse_callable({ tarr, i, end }));
                    continue;
                case TokenType::KW_INTR:
                    i++;
                    end = tarr.search(LineEndCriteria(0), i);
                    if (end == i)
                        throw SyntaxError(tarr[i], "Expected identifier after 'intr'");
                    if (end < 0)
                        end = tarr.size();
                    ast_module.intrs.push_back(parse_callable({ tarr, i, end }));
                    continue;
                default:
                    throw SyntaxError(tarr[i], "Invalid token");
                }
            }

            return ast_module;
        }
    }
}
