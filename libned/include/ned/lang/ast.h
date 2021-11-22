#ifndef NED_AST_H
#define NED_AST_H

#include <vector>
#include <string>

#include <ned/lang/lexer.h>

namespace nn
{
    namespace lang
    {
        enum class LineType
        {
            INVALID,
            IF,
            ELIF,
            ELSE,
            RAISE,
            PRINT,
            EXPORT,
            WHILE,
            FOR,
            BREAK,
            CONTINUE,
            DECL,
            EXPR
        };

        // Generic base class for all types of lines of code
        struct AstLine
        {
            LineType ty;
        };

        // Declarations in a struct or fn sequence
        struct AstRegDecl
        {

        };

        // Carg parameters allowed in a AstArgDecl 
        struct AstArgExpr
        {

        };

        // Standard argument declaration - function varg or any carg
        // Not allowed to have any tensor declarations
        // Packing is allowed
        struct AstArgDecl
        {

        };

        // Special argument declarations - def/intr varg
        // Only tensor declarations are allowed
        // Packing is allowed
        struct AstVargDecl
        {

        };

        // Top level function definition
        struct AstFn
        {
            std::string name;

        };

        // Top level block definition
        struct AstDef
        {

        };

        // Top level intrinsic definition
        struct AstIntr
        {
            std::string name;
            std::vector<AstArgDecl> cargs;
            std::vector<AstVargDecl> vargs;
            std::vector<AstLine*> lines;
        };

        // Top level structure definition
        struct AstStruct
        {
            std::string name;
            std::vector<AstArgDecl> cargs;
            std::vector<AstRegDecl> decls;
        };

        struct AstImport
        {
            std::vector<std::string> imp;
        };

        struct AstModule
        {
            std::string fname;
            std::vector<AstImport> imports;
            std::vector<AstStruct> structs;
            std::vector<AstFn> funcs;
            std::vector<AstDef> defs;
            std::vector<AstIntr> intrs;
        };

        AstRegDecl  parse_reg_decl  (const TokenArray& tarr);
        AstLine*    parse_line      (const TokenArray& tarr);

        AstArgDecl  parse_arg_decl  (const TokenArray& tarr);
        AstVargDecl parse_varg_decl (const TokenArray& tarr);

        AstFn       parse_fn        (const TokenArray& tarr);
        AstDef      parse_def       (const TokenArray& tarr);
        AstIntr     parse_intr      (const TokenArray& tarr);
        AstStruct   parse_struct    (const TokenArray& tarr);
        AstImport   parse_import    (const TokenArray& tarr);
        AstModule   parse_module    (const TokenArray& tarr);
    }
}

#endif
