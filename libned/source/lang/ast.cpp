#include <ned/lang/ast.h>
#include <ned/lang/obj.h>

#include <functional>
#include <cassert>

// Parses comma deliminated token sequences within a matching set of brackets
// Returns the index in tarr immediately after the CLOSE token was found
template<typename T, std::function<T(const nn::lang::TokenArray&)> parse_fn, nn::lang::TokenType OPEN, nn::lang::TokenType CLOSE>
inline int parse_args(int i, std::vector<T>& args, const nn::lang::TokenArray& tarr)
{
    assert(tarr[0]->ty == OPEN);
    
    // Advancing one token past the opening bracket
    i++;
    int end = tarr.search(nn::lang::CargEndCriteria(), i);
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
        end = tarr.search(nn::lang::CargEndCriteria(), i);
        args.push_back(parse_fn({ tarr, i, end }));
        i = end + 1;
    } while (tarr[end]->ty == nn::lang::TokenType::COMMA);

    return i;
}

// Parses lines from a token sequence
template<typename T, T(*parse_fn)(const nn::lang::TokenArray&)>
inline void parse_lines(int i, int indent_level, std::vector<T>& lines, const nn::lang::TokenArray& tarr)
{
    int end;
    do
    {
        end = tarr.search(nn::lang::LineEndCriteria(indent_level), i);
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

namespace nn
{
    namespace lang
    {
        AstIntr parse_intr(const TokenArray& tarr)
        {
            AstIntr ast_intr;

            // Setup and getting the name of the intr definition
            int i = 0;
            for (; tarr[i]->is_whitespace(); i++);
            ast_intr.name = tarr[i]->get<TokenType::IDN>().val;

            // Parsing out the cargs from the intr definition
            for (; tarr[i]->is_whitespace(); i++);
            if (tarr[i]->ty == TokenType::ANGLE_O)
            {
                i = parse_args<AstArgDecl, parse_arg_decl, TokenType::ANGLE_O, TokenType::ANGLE_C>(i, ast_intr.cargs, tarr);
                for (; tarr[i]->is_whitespace(); i++);
            }

            // Parsing out the vargs from the intr definition
            if (tarr[i]->ty == TokenType::ROUND_O)
            {
                i = parse_args<AstVargDecl, parse_varg_decl, TokenType::ROUND_O, TokenType::ROUND_C>(i, ast_intr.vargs, tarr);
                for (; tarr[i]->is_whitespace(); i++);
            }

            // Finding the ':' that starts the struct body
            tarr[i]->get<TokenType::COLON>();
            for (; tarr[i]->ty == TokenType::INDENT; i++);  // one token past the first endl after the ':'
            tarr[i]->get<TokenType::ENDL>();
            parse_lines<AstLine*, parse_line>(i, 1, ast_intr.lines, tarr);

            return ast_intr;
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
            parse_lines<AstRegDecl, parse_reg_decl>(i, 1, ast_struct.decls, tarr);

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
                    ast_module.funcs.push_back(parse_fn({ tarr, i, end }));
                    continue;
                case TokenType::KW_DEF:
                    i++;
                    end = tarr.search(LineEndCriteria(0), i);
                    if (end == i)
                        throw SyntaxError(tarr[i], "Expected identifier after 'def'");
                    if (end < 0)
                        end = tarr.size();
                    ast_module.defs.push_back(parse_def({ tarr, i, end }));
                    continue;
                case TokenType::KW_INTR:
                    i++;
                    end = tarr.search(LineEndCriteria(0), i);
                    if (end == i)
                        throw SyntaxError(tarr[i], "Expected identifier after 'intr'");
                    if (end < 0)
                        end = tarr.size();
                    ast_module.intrs.push_back(parse_intr({ tarr, i, end }));
                    continue;
                default:
                    throw SyntaxError(tarr[i], "Invalid token");
                }
            }

            return ast_module;
        }
    }
}
