#ifndef NED_AST_H
#define NED_AST_H

#include <vector>
#include <string>

#include <ned/lang/lexer.h>

namespace nn
{
    namespace lang
    {
        enum class ExprType
        {
            INVALID,
            UNARY_NOT,
            UNARY_POS,
            UNARY_NEG,
            UNARY_UNPACK,
            BINARY_ADD,
            BINARY_SUB,
            BINARY_MUL,
            BINARY_DIV,
            BINARY_IADD,
            BINARY_ISUB,
            BINARY_IMUL,
            BINARY_IDIV,
            BINARY_ASSIGN,
            BINARY_CMP_EQ,
            BINARY_CMP_NE,
            BINARY_CMP_GT,
            BINARY_CMP_LT,
            BINARY_CMP_GE,
            BINARY_CMP_LE,
            DOT,
            CARGS_CALL,
            VARGS_CALL
        };

        class AstExpr
        {
        public:
            virtual void codegen() = 0;
        };

        class AstExprUnaryNot :
            public AstExpr
        {
        public:
            virtual void codegen() = 0;
        };

        class AstExprBoolLit :
            public AstExpr
        {
            bool val;
        public:
            AstExprBoolLit(bool val) : val(val) {}
            virtual void codegen() = 0;
        };

        class AstExprIntLit :
            public AstExpr
        {
            int64_t val;
        public:
            AstExprIntLit(int64_t val) : val(val) {}
            virtual void codegen() = 0;
        };

        struct AstExprFloatLit
        {
            double val;
        public:
            AstExprFloatLit(double val) : val(val) {}
            virtual void codegen() = 0;
        };

        struct AstExprStrLit
        {
            std::string val;
        };

        // Linear aggregate types (array, tuple)
        struct AstExprAggLit
        {
            std::vector<AstExpr> elems;
        };

        struct AstExprUnaryOp
        {
            AstExpr* expr;
        };

        struct AstExprBinaryOp
        {
            AstExpr* left;
            AstExpr* right;
        };

        struct AstExprDot
        {
            AstExpr* expr;
            std::string val;
        };

        struct AstExprCall
        {
            AstExpr* callee;
            std::vector<AstExpr> args;
        };

        struct AstExpr
        {
            ExprType ty = ExprType::INVALID;
            union
            {
                AstExprBoolLit expr_bool;
                AstExprIntLit expr_int;
                AstExprFloatLit expr_float;
                AstExprStrLit expr_string;
                AstExprAggLit expr_agg;
                AstExprUnaryOp expr_unary;
                AstExprBinaryOp expr_binary;
                AstExprDot expr_dot;
                AstExprCall expr_call;
            };
        };

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
        struct AstLineBuiltinFunc
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
            AstLineDecl decl;
            AstExpr iter;
            std::vector<AstLine> body;
        };

        // Line containing only a declaration
        struct AstLineDecl
        {
            AstExpr type;
            std::vector<AstExpr> cargs;
            std::string name;
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
                AstLineExport       line_export;
                AstLineBuiltinFunc  line_func;
                AstLineBranch       line_branch;
                AstLineElse         line_else;
                AstLineFor          line_for;
                AstLineDecl         line_decl;
                AstLineExpr         line_expr;
            };
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
            std::vector<AstLineDecl> decls;
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

        
        AstExpr*     parse_expr_alloc (const TokenArray& tarr);
        AstExpr      parse_expr       (const TokenArray& tarr);

        AstLineDecl  parse_line_decl  (const TokenArray& tarr);
        AstLine*     parse_line_alloc (const TokenArray& tarr);
        AstLine      parse_line       (const TokenArray& tarr);

        AstArgDecl   parse_arg_decl   (const TokenArray& tarr);

        AstCallable  parse_callable   (const TokenArray& tarr);
        AstStruct    parse_struct     (const TokenArray& tarr);
        AstImport    parse_import     (const TokenArray& tarr);
        AstModule    parse_module     (const TokenArray& tarr);
    }
}

#endif
