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
            BINARY_IDX,
            DOT,
            VAR_DECL,
            CARGS_CALL,
            VARGS_CALL,
            FN_DECL,
            DEF_DECL,
            KW,
            VAR
        };

        // Code Block signatures

        struct AstArgDecl
        {
            bool is_packed = false;
            AstExpr type_expr;  // The expression that was used to define the pased type
            std::string var_name;
        };

        // Top level structure definition (struct)
        struct AstStructSig
        {
            std::string name;
            std::vector<AstArgDecl> cargs;
        };

        // Top level function definition (fn)
        struct AstFnSig
        {
            std::string name;
            std::vector<AstArgDecl> cargs;
            std::vector<AstArgDecl> vargs;
            std::vector<AstExpr> rets;
        };

        // Top level block / intrinsic definition (def/intr)
        struct AstBlockSig
        {
            std::string name;
            std::vector<AstArgDecl> cargs;
            std::vector<AstArgDecl> vargs;
            std::vector<std::string> rets;
        };

        enum class ExprKW
        {
            TYPE,
            VAR,
            FP,
            BOOL,
            INT,
            FLOAT,
            STR,
            ARRAY,
            TUPLE,
            F16,
            F32,
            F64
        };

        // Linear aggregate types (array, tuple)
        struct AstExprAggLit
        {
            std::vector<AstExpr> elems = {};
        };

        struct AstExprUnaryOp
        {
            AstExpr* expr = nullptr;

            ~AstExprUnaryOp();
        };

        struct AstExprBinaryOp
        {
            AstExpr* left;
            AstExpr* right;

            ~AstExprBinaryOp();
        };

        struct AstExprName
        {
            AstExpr* expr = nullptr;
            std::string val = "";

            ~AstExprName();
        };

        struct AstExprCall
        {
            AstExpr* callee = nullptr;
            std::vector<AstExpr> args = {};

            ~AstExprCall();
        };

        struct AstExpr
        {
            std::string fname;
            uint32_t line_start;
            uint32_t line_end;
            uint32_t col_start;
            uint32_t col_end;

            ExprType ty = ExprType::INVALID;
            union
            {
                bool expr_bool;
                ExprKW expr_kw;
                int64_t expr_int;
                double expr_float;
                std::string expr_string;
                AstExprAggLit expr_agg;
                AstExprUnaryOp expr_unary;
                AstExprBinaryOp expr_binary;
                AstExprName expr_name;
                AstExprCall expr_call;
                AstFnSig expr_fn_decl;
                AstBlockSig expr_def_decl;
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
            std::string fname;
            uint32_t line_start;
            uint32_t line_end;
            uint32_t col_start;
            uint32_t col_end;

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

            AstLine() {}
            ~AstLine();
        };

        template<typename T>
        concept CodeBlockSig =
            std::is_same<T, AstStructSig>::value ||
            std::is_same<T, AstFnSig>    ::value ||
            std::is_same<T, AstBlockSig> ::value ;

        // Top level block of code - contains a signature and body
        template<CodeBlockSig SIG>
        struct AstCodeBlock
        {
            SIG signature;
            std::vector<AstLine> body;
        };

        using AstStruct = AstCodeBlock<AstStructSig>;
        using AstFn     = AstCodeBlock<AstFnSig    >;
        using AstBlock  = AstCodeBlock<AstBlockSig >;

        struct AstImport
        {
            std::vector<std::string> imp;
        };

        struct AstModule
        {
            std::string fname;
            std::vector<AstImport> imports;
            std::vector<AstStruct> structs;
            std::vector<AstFn>     funcs;
            std::vector<AstBlock>  defs;
            std::vector<AstBlock>  intrs;
        };

        // parse_* functions return true on failure, false on success

        bool parse_expr       (Errors& errs, const TokenArray& tarr, AstExpr&);

        bool parse_line       (Errors& errs, const TokenArray& tarr, AstLine&, int indent_level);

        bool parse_arg_decl   (Errors& errs, const TokenArray& tarr, AstArgDecl&);

        bool parse_struct_sig (Errors& errs, const TokenArray& tarr, AstStructSig&);

        template<CodeBlockSig SIG> bool parse_signature  (Errors& errs, const TokenArray& tarr, SIG& sig);
        template<CodeBlockSig SIG> bool parse_code_block (Errors& errs, const TokenArray& tarr, AstCodeBlock<SIG>&);

        bool parse_import     (Errors& errs, const TokenArray& tarr, AstImport&);
        bool parse_module     (Errors& errs, const TokenArray& tarr, AstModule&);
    }
}

#endif
