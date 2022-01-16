#ifndef NED_AST_H
#define NED_AST_H

#include <ned/lang/lexer.h>

#include <vector>
#include <string>
#include <memory>

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
            std::unique_ptr<AstExpr> type_expr;  // The expression that was used to define the passed type
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
            std::vector<AstExpr> elems;
        };

        struct AstExprUnaryOp
        {
            std::unique_ptr<AstExpr> expr;
        };

        struct AstExprBinaryOp
        {
            std::unique_ptr<AstExpr> left;
            std::unique_ptr<AstExpr> right;
        };

        struct AstExprName
        {
            std::unique_ptr<AstExpr> expr;
            std::string val = "";
        };

        struct AstExprCall
        {
            std::unique_ptr<AstExpr> callee;
            std::vector<AstExpr> args = {};

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
            AstExpr(AstExpr&&);
            AstExpr& operator=(AstExpr&&);
            ~AstExpr();
        };

        struct AstLine;
        enum class LineType
        {
            INVALID,
            BREAK,
            CONTINUE,
            EXPORT,
            EXTERN,
            RAISE,
            PRINT,
            RETURN,
            IF,
            ELIF,
            ELSE,
            WHILE,
            FOR,
            EXPR,
            FORWARD,
            BACKWARD,
            EVALMODE
        };

        // An export statement needs a name associated with it
        struct AstLineExport
        {
            std::string var_name;
        };

        struct AstLineExtern
        {
            std::string var_name;
            AstExpr init_expr;
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

        // eval mode labeled block
        struct AstLineLabel
        {
            std::string label;
            std::vector<AstLine> body;
        };

        // else / forward / backward statement
        struct AstLineBlock
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
                AstLineExtern     line_extern;
                AstLineUnaryFunc  line_func;
                AstLineBranch     line_branch;
                AstLineBlock      line_block;
                AstLineFor        line_for;
                AstLineExpr       line_expr;
                AstLineLabel      line_label;
            };

            AstLine() {}
            AstLine(AstLine&& line);
            AstLine& operator=(AstLine&& line);
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

        struct AstInit
        {
            std::string name;
            std::vector<AstArgDecl> cargs;
        };

        struct AstNamespace
        {
            std::string name;
            std::vector<AstNamespace> namespaces;
            std::vector<AstStruct>    structs;
            std::vector<AstFn>        funcs;
            std::vector<AstBlock>     defs;
            std::vector<AstBlock>     intrs;
            std::vector<AstInit>      inits;
        };

        struct AstModule
        {
            std::string fname;
            std::vector<AstImport>    imports;
            std::vector<AstNamespace> namespaces;
            std::vector<AstStruct>    structs;
            std::vector<AstFn>        funcs;
            std::vector<AstBlock>     defs;
            std::vector<AstBlock>     intrs;
            std::vector<AstInit>      inits;
        };

        // parse_* functions return true on failure, false on success

        bool parse_expr       (const TokenArray& tarr, AstExpr&);

        bool parse_line       (const TokenArray& tarr, AstLine&, int indent_level);

        bool parse_arg_decl   (const TokenArray& tarr, AstArgDecl&);

        bool parse_struct_sig (const TokenArray& tarr, AstStructSig&);

        template<CodeBlockSig SIG> bool parse_signature  (const TokenArray& tarr, SIG& sig);
        template<CodeBlockSig SIG> bool parse_code_block (const TokenArray& tarr, AstCodeBlock<SIG>&, int indent_level);

        bool parse_namespace  (const TokenArray& tarr, AstNamespace&, int indent_level);
        bool parse_import     (const TokenArray& tarr, AstImport&);
        bool parse_init       (const TokenArray& tarr, AstInit&);
        bool parse_module     (const TokenArray& tarr, AstModule&);
    }
}

#endif
