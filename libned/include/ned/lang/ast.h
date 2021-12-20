#ifndef NED_AST_H
#define NED_AST_H

#include <ned/lang/lexer.h>
#include <ned/lang/errors.h>

#include <vector>
#include <string>

namespace nn
{
    namespace lang
    {
        struct AstExpr;
        enum class ExprType
        {
            INVALID,
            LIT_BOOL,
            LIT_INT,
            LIT_FLOAT,
            LIT_STRING,
            LIT_ARRAY,
            LIT_TUPLE,
            UNARY_POS,
            UNARY_NEG,
            UNARY_NOT,
            UNARY_UNPACK,
            BINARY_ADD,
            BINARY_SUB,
            BINARY_MUL,
            BINARY_DIV,
            BINARY_MOD,
            BINARY_IADD,
            BINARY_ISUB,
            BINARY_IMUL,
            BINARY_IDIV,
            BINARY_IMOD,
            BINARY_ASSIGN,
            BINARY_AND,
            BINARY_OR,
            BINARY_CMP_EQ,
            BINARY_CMP_NE,
            BINARY_CMP_GT,
            BINARY_CMP_LT,
            BINARY_CMP_GE,
            BINARY_CMP_LE,
            DOT,
            CARGS_CALL,
            VARGS_CALL,
            DECL
        };

        // Linear aggregate types (array, tuple)
        struct AstExprAggLit
        {
            std::vector<AstExpr> elems;
        };

        struct AstExprUnaryOp
        {
            AstExpr* expr;

            ~AstExprUnaryOp();
        };

        struct AstExprBinaryOp
        {
            AstExpr* left;
            AstExpr* right;

            ~AstExprBinaryOp();
        };

        struct AstExprDot
        {
            AstExpr* expr;
            std::string val;

            ~AstExprDot();
        };

        struct AstExprCall
        {
            AstExpr* callee;
            std::vector<AstExpr> args;

            ~AstExprCall();
        };

        struct AstExprDecl
        {
            AstExpr* type;
            std::vector<AstExpr> cargs;
            std::string name;

            ~AstExprDecl();
        };

        struct AstExpr
        {
            ExprType ty = ExprType::INVALID;
            union
            {
                bool expr_bool;
                int64_t expr_int;
                double expr_float;
                std::string expr_string;
                AstExprAggLit expr_agg;
                AstExprUnaryOp expr_unary;
                AstExprBinaryOp expr_binary;
                AstExprDot expr_dot;
                AstExprCall expr_call;
                AstExprDecl expr_decl;
            };

            AstExpr() {}
            ~AstExpr();
        };

        struct AstLine;
        enum class LineType
        {
            INVALID,
            BREAK,
            CONTINUE,
            EXPORT,
            RAISE,
            PRINT,
            RETURN,
            IF,
            ELIF,
            ELSE,
            WHILE,
            FOR,
            DECL,
            EXPR
        };

        // An export statement needs a name associated with it
        struct AstLineExport
        {
            std::string var_name;
        };

        // raise / return / print statement
        struct AstLineUnaryFunc
        {
            AstExpr expr;
        };

        // if / elif statement and while loop
        struct AstLineBranch
        {
            AstExpr cond;
            std::vector<AstLine> body;
        };

        // else statement
        struct AstLineElse
        {
            std::vector<AstLine> body;
        };

        // for loop
        struct AstLineFor
        {
            AstExpr decl;
            AstExpr iter;
            std::vector<AstLine> body;
        };

        // Line containing an arbitrary expression
        struct AstLineExpr
        {
            AstExpr line;
        };

        // Generic base class for all types of lines of code
        struct AstLine
        {
            // This thing's value will determine which union element will be accessed
            LineType ty = LineType::INVALID;
            union
            {
                AstLineExport     line_export;
                AstLineUnaryFunc  line_func;
                AstLineBranch     line_branch;
                AstLineElse       line_else;
                AstLineFor        line_for;
                AstLineExpr       line_expr;
            };

            ~AstLine();
        };

        // Argument declaration - varg or carg
        // Packing is allowed
        struct AstArgDecl
        {
            bool is_packed = false;
            AstExpr type_expr;  // The expression that was used to define the pased type
            std::string var_name;
        };

        // Top level function / block / intrinsic definition
        struct AstCallable
        {
            std::string name;
            std::vector<AstArgDecl> cargs;
            std::vector<AstArgDecl> vargs;
            std::vector<AstLine> lines;
        };

        // Top level structure definition
        struct AstStruct
        {
            std::string name;
            std::vector<AstArgDecl> cargs;
            std::vector<AstLine> decls;
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
            std::vector<AstCallable> funcs;
            std::vector<AstCallable> defs;
            std::vector<AstCallable> intrs;
        };

        // parse_* functions return true on failure, false on success

        bool parse_expr       (ParsingErrors& errs, const TokenArray& tarr, AstExpr&);

        bool parse_line       (ParsingErrors& errs, const TokenArray& tarr, AstLine&, int indent_level);

        bool parse_arg_decl   (ParsingErrors& errs, const TokenArray& tarr, AstArgDecl&);

        bool parse_callable   (ParsingErrors& errs, const TokenArray& tarr, AstCallable&);
        bool parse_struct     (ParsingErrors& errs, const TokenArray& tarr, AstStruct&);
        bool parse_import     (ParsingErrors& errs, const TokenArray& tarr, AstImport&);
        bool parse_module     (ParsingErrors& errs, const TokenArray& tarr, AstModule&);
    }
}

#endif
