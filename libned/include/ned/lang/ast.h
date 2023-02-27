#ifndef NED_AST_H
#define NED_AST_H

#include <ned/lang/lexer.h>

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <map>
#include <array>

namespace nn
{
    namespace lang
    {
        using PrecTable = std::array<uint8_t, (size_t)TokenType::TOKEN_TYPE_END>;
        using RtolTable = std::array<bool, 11>;

        struct AstNodeInfo
        {
            std::string fname = "";
            uint32_t line_start = 0;
            uint32_t line_end = 0;
            uint32_t col_start = 0;
            uint32_t col_end = 0;
        };

        struct AstExpr;
        struct AstLine;

        enum class ExprKW
        {
            NUL,
            TYPE,
            VOID,
            INIT,
            FTY,
            BOOL,
            INT,
            FLOAT,
            STR,
            CFG,
            ARRAY,
            TUPLE,
            F16,
            F32,
            F64
        };

        std::string to_string(ExprKW kw);

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

        struct AstExprIndex
        {
            struct Elem
            {
                enum class Type
                {
                    INVALID,
                    ELLIPSES,
                    DIRECT,
                    SLICE
                } ty = Type::INVALID;

                std::unique_ptr<AstExpr> lhs = nullptr;
                std::unique_ptr<AstExpr> rhs = nullptr;
            };

            std::unique_ptr<AstExpr> expr;
            std::vector<Elem> args;
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

        struct AstEnumEntry
        {
            AstNodeInfo node_info;
            std::string name;
            std::vector<AstLine> lines;
        };

        struct AstExprEnum
        {
            AstCargSig signature;
            std::vector<AstEnumEntry> entries;
            // Technically, this is redunant info, but it'll speed up compilation
            std::unordered_map<std::string, const AstEnumEntry*> entry_map;

            AstExprEnum() {}
            AstExprEnum(AstExprEnum&& ast_enum) noexcept
            {
                // fuck you msvc
                signature = std::move(ast_enum.signature);
                entries = std::move(ast_enum.entries);
                entry_map = std::move(ast_enum.entry_map);
            }
        };

        struct AstExprNamespace
        {
            std::vector<AstExpr> lines;
        };

        // Carg only definition (struct/enum/init)
        struct AstCargSig
        {
            std::vector<AstExpr> cargs;
        };

        // Function definition (fn)
        struct AstFnSig
        {
            std::vector<AstExpr> cargs;
            std::vector<AstExpr> vargs;
            std::vector<AstExpr> rets;
        };

        // Block / intrinsic definition (def/intr)
        struct AstBlockSig
        {
            std::vector<AstExpr> cargs;
            std::vector<AstExpr> vargs;
            std::vector<std::string> rets;
        };

        struct AstExprStruct
        {
            AstCargSig signature;
            std::vector<AstExpr> body;
        };

        struct AstExprBlock
        {
            AstBlockSig signature;
            bool is_bytecode;
            std::vector<AstExpr> body;
            TokenArray tarr = {};
        };

        struct AstExprFn
        {
            AstFnSig signature;
            bool is_bytecode;
            std::vector<AstExpr> body;
            TokenArray tarr = {};
        };

        struct AstExprInit
        {
            AstCargSig signature;
            bool is_bytecode;
            std::vector<AstExpr> body;
            TokenArray tarr = {};
        };

        struct AstExprImport
        {
            std::vector<std::string> imp;
        };

        struct AstExpr
        {
            enum class Type
            {
                INVALID,
                EMPTY,
                KW,
                VAR,
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
                UNARY_MUT,
                UNARY_REF,
                BINARY_ADD,
                BINARY_SUB,
                BINARY_MUL,
                BINARY_DIV,
                BINARY_MOD,
                BINARY_POW,
                BINARY_IADD,
                BINARY_ISUB,
                BINARY_IMUL,
                BINARY_IDIV,
                BINARY_IMOD,
                BINARY_IPOW,
                BINARY_ASSIGN,
                BINARY_AND,
                BINARY_OR,
                BINARY_CMP_EQ,
                BINARY_CMP_NE,
                BINARY_CMP_GT,
                BINARY_CMP_LT,
                BINARY_CMP_GE,
                BINARY_CMP_LE,
                BINARY_CAST,
                BINARY_DECL,
                INDEX,
                DOT,
                CALL_CARGS,
                CALL_VARGS,
                DEFN_NAMESPACE,
                DEFN_ENUM,
                DEFN_STRUCT,
                DEFN_DEF,
                DEFN_INTR,
                DEFN_FN,
                DEFN_INIT,
                DECL_DEF,
                DECL_INTR,
                DECL_FN,
                DECL_INIT,
                IMPORT,
            } ty = Type::INVALID;
            AstNodeInfo node_info;

            union
            {
                bool             expr_bool;
                ExprKW           expr_kw;
                int64_t          expr_int;
                double           expr_float;
                std::string      expr_string;
                AstExprAggLit    expr_agg;
                AstExprUnaryOp   expr_unary;
                AstExprBinaryOp  expr_binary;
                AstExprIndex     expr_index;
                AstExprName      expr_name;
                AstExprCall      expr_call;
                AstExprNamespace expr_namespace;
                AstExprEnum      expr_enum;
                AstExprStruct    expr_struct;
                AstExprBlock     expr_block;
                AstExprFn        expr_fn;
                AstExprInit      expr_init;
                AstBlockSig      expr_block_decl;
                AstFnSig         expr_fn_decl;
                AstCargSig       expr_init_decl;
                AstExprImport    expr_import;
            };

            AstExpr();
            AstExpr(AstExpr&& expr) noexcept;             // results in expr.ty <- Type::INVALID
            AstExpr& operator=(AstExpr&& expr) noexcept;  // results in expr.ty <- Type::INVALID
            ~AstExpr();

        private:
            void reset() noexcept;
        };

        struct AstLine;
        enum class LineType
        {
            INVALID,
            BREAK,
            CONTINUE,
            EXPORT,
            EXTERN,
            CFGINFO,
            RETURN,
            MATCH,
            IF,
            ELIF,
            WHILE,
            ELSE,
            EVALMODE,
            FOR,
            EXPR,
        };

        struct AstMatchElem
        {
            AstNodeInfo node_info;

            std::string label;
            std::vector<AstLine> body;
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

        // return statement
        struct AstLineUnaryFunc
        {
            AstExpr expr;
        };

        // __add_cfg_info statement
        struct AstLineCfgInfo
        {
            AstExpr name_expr;
            AstExpr cfg_expr;
        };

        // match statement
        struct AstLineMatch
        {
            AstExpr arg;
            std::vector<AstMatchElem> elems;
        };

        // if / elif statement and while loop
        struct AstLineBranch
        {
            AstExpr cond;
            std::vector<AstLine> body;
        };

        // else statement
        struct AstLineBlock
        {
            std::vector<AstLine> body;
        };

        // eval mode labeled block
        struct AstLineLabel
        {
            std::string label;
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
            AstNodeInfo node_info;

            // This thing's value will determine which union element will be accessed
            LineType ty = LineType::INVALID;
            union
            {
                AstLineExport     line_export;
                AstLineExtern     line_extern;
                AstLineUnaryFunc  line_func;
                AstLineCfgInfo    line_cfginfo;
                AstLineMatch      line_match;
                AstLineBranch     line_branch;
                AstLineBlock      line_block;
                AstLineLabel      line_label;
                AstLineFor        line_for;
                AstLineExpr       line_expr;
            };

            AstLine();
            AstLine(AstLine&& line) noexcept;
            AstLine& operator=(AstLine&& line) noexcept;
            ~AstLine();

        private:
            void do_move(AstLine&& line) noexcept;
        };

        struct AstModule
        {
            std::string fname;
            std::vector<AstExpr> lines;
        };

        /*
        *
        * | Precedence |   Operators   | Associativity |
        * | ---------- | ------------- | ------------- |
        * | 11.        | expr()        | left-to-right |
        * |            | expr<>        |               |
        * |            | expr[]        |               |
        * |            | expr.idn      |               |
        * | ---------- | ------------- | ------------- |
        * | 10.        | namespace     | n/a           |
        * |            | struct        |               |
        * |            | enum          |               |
        * |            | def           |               |
        * |            | intr          |               |
        * |            | fn            |               |
        * |            | init          |               |
        * |            | import        |               |
        * |            | + (unary)     |               |
        * |            | - (unary)     |               |
        * |            | * (unary)     |               |
        * |            | not           |               |
        * |            | mut           |               |
        * |            | ref           |               |
        * | ---------- | ------------- | ------------- |
        * | 9.         | ::            | right-to-left |
        * | ---------- | ------------- | ------------- |
        * | 8.         | :             | right-to-left |
        * | ---------- | ------------- | ------------- |
        * | 7.         | ^             | left-to-right |
        * | ---------- | ------------- | ------------- |
        * | 6.         | *             | left-to-right |
        * |            | /             |               |
        * |            | %             |               |
        * | ---------- | ------------- | ------------- |
        * | 5.         | +             | left-to-right |
        * |            | -             |               |
        * | ---------- | ------------- | ------------- |
        * | 4.         | ==            | left-to-right |
        * |            | !=            |               |
        * |            | <             |               |
        * |            | >             |               |
        * |            | <=            |               |
        * |            | >=            |               |
        * | ---------- | ------------- | ------------- |
        * | 3.         | and           | left-to-right |
        * |            | or            |               |
        * | ---------- | ------------- | ------------- |
        * | 2.         | ,             | n/a           |
        * | ---------- | ------------- | ------------- |
        * | 1.         | =             | right-to-left |
        * |            | +=            |               |
        * |            | -=            |               |
        * |            | *=            |               |
        * |            | /=            |               |
        * |            | %=            |               |
        * |            | ^=            |               |
        *
        */

        constexpr auto default_prec_table = []() {
            PrecTable ret = {};

            ret[(size_t)TokenType::KW_NAMESPACE] = 10;
            ret[(size_t)TokenType::KW_STRUCT   ] = 10;
            ret[(size_t)TokenType::KW_ENUM     ] = 10;
            ret[(size_t)TokenType::KW_DEF      ] = 10;
            ret[(size_t)TokenType::KW_INTR     ] = 10;
            ret[(size_t)TokenType::KW_FN       ] = 10;
            ret[(size_t)TokenType::KW_INIT     ] = 10;
            ret[(size_t)TokenType::KW_IMPORT   ] = 10;
            ret[(size_t)TokenType::CAST        ] = 9;
            ret[(size_t)TokenType::COLON       ] = 8;
            ret[(size_t)TokenType::POW         ] = 7;
            ret[(size_t)TokenType::STAR        ] = 6;
            ret[(size_t)TokenType::DIV         ] = 6;
            ret[(size_t)TokenType::MOD         ] = 6;
            ret[(size_t)TokenType::ADD         ] = 5;
            ret[(size_t)TokenType::SUB         ] = 5;
            ret[(size_t)TokenType::CMP_EQ      ] = 4;
            ret[(size_t)TokenType::CMP_NE      ] = 4;
            ret[(size_t)TokenType::CMP_GT      ] = 4;
            ret[(size_t)TokenType::CMP_LT      ] = 4;
            ret[(size_t)TokenType::CMP_GE      ] = 4;
            ret[(size_t)TokenType::CMP_LE      ] = 4;
            ret[(size_t)TokenType::KW_AND      ] = 3;
            ret[(size_t)TokenType::KW_OR       ] = 3;
            ret[(size_t)TokenType::COMMA       ] = 2;
            ret[(size_t)TokenType::ASSIGN      ] = 1;
            ret[(size_t)TokenType::IADD        ] = 1;
            ret[(size_t)TokenType::ISUB        ] = 1;
            ret[(size_t)TokenType::IMUL        ] = 1;
            ret[(size_t)TokenType::IDIV        ] = 1;
            ret[(size_t)TokenType::IMOD        ] = 1;
            ret[(size_t)TokenType::IPOW        ] = 1;


            return ret;
        }();

        constexpr auto args_prec_table = []() {
            PrecTable ret = {};

            ret[(size_t)TokenType::KW_NAMESPACE] = 10;
            ret[(size_t)TokenType::KW_STRUCT   ] = 10;
            ret[(size_t)TokenType::KW_ENUM     ] = 10;
            ret[(size_t)TokenType::KW_DEF      ] = 10;
            ret[(size_t)TokenType::KW_INTR     ] = 10;
            ret[(size_t)TokenType::KW_FN       ] = 10;
            ret[(size_t)TokenType::KW_INIT     ] = 10;
            ret[(size_t)TokenType::KW_IMPORT   ] = 10;
            ret[(size_t)TokenType::CAST        ] = 9;
            ret[(size_t)TokenType::COLON       ] = 8;
            ret[(size_t)TokenType::POW         ] = 7;
            ret[(size_t)TokenType::STAR        ] = 6;
            ret[(size_t)TokenType::DIV         ] = 6;
            ret[(size_t)TokenType::MOD         ] = 6;
            ret[(size_t)TokenType::ADD         ] = 5;
            ret[(size_t)TokenType::SUB         ] = 5;
            ret[(size_t)TokenType::CMP_EQ      ] = 4;
            ret[(size_t)TokenType::CMP_NE      ] = 4;
            ret[(size_t)TokenType::CMP_GT      ] = 4;
            ret[(size_t)TokenType::CMP_LT      ] = 4;
            ret[(size_t)TokenType::CMP_GE      ] = 4;
            ret[(size_t)TokenType::CMP_LE      ] = 4;
            ret[(size_t)TokenType::KW_AND      ] = 3;
            ret[(size_t)TokenType::KW_OR       ] = 3;
            ret[(size_t)TokenType::ASSIGN      ] = 2;
            ret[(size_t)TokenType::IADD        ] = 2;
            ret[(size_t)TokenType::ISUB        ] = 2;
            ret[(size_t)TokenType::IMUL        ] = 2;
            ret[(size_t)TokenType::IDIV        ] = 2;
            ret[(size_t)TokenType::IMOD        ] = 2;
            ret[(size_t)TokenType::IPOW        ] = 2;
            ret[(size_t)TokenType::COMMA       ] = 1;

            return ret;
        }();

        constexpr RtolTable default_rtol_table = []() {
            RtolTable ret = {};

            ret[(size_t)11 - 1] = false ;
            ret[(size_t) 9 - 1] = true  ;
            ret[(size_t) 8 - 1] = true  ;
            ret[(size_t) 7 - 1] = false ;
            ret[(size_t) 6 - 1] = false ;
            ret[(size_t) 5 - 1] = false ;
            ret[(size_t) 4 - 1] = false ;
            ret[(size_t) 3 - 1] = false ;
            ret[(size_t) 1 - 1] = true  ;

            return ret;
        }();

        constexpr auto args_rtol_table = []() {
            RtolTable ret = {};

            ret[(size_t)11 - 1] = false ;
            ret[(size_t) 9 - 1] = true  ;
            ret[(size_t) 8 - 1] = true  ;
            ret[(size_t) 7 - 1] = false ;
            ret[(size_t) 6 - 1] = false ;
            ret[(size_t) 5 - 1] = false ;
            ret[(size_t) 4 - 1] = false ;
            ret[(size_t) 3 - 1] = false ;
            ret[(size_t) 2 - 1] = true  ;

            return ret;
        }();

        // parse_* functions return true on failure, false on success

        bool parse_match_elem (const TokenArray& tarr, AstMatchElem&, int indent_level);

        bool parse_signature  (const TokenArray& tarr, AstCargSig  & sig, int indent_level);
        bool parse_signature  (const TokenArray& tarr, AstBlockSig & sig, int indent_level);
        bool parse_signature  (const TokenArray& tarr, AstFnSig    & sig, int indent_level);

        bool parse_line       (const TokenArray& tarr, AstLine& ast_line, int indent_level);
        template<PrecTable prec_table, RtolTable rtol_table>
        int  parse_subexpr    (const TokenArray& tarr, AstExpr& ast_expr, int indent_level, int prec);
        template<PrecTable prec_table = default_prec_table, RtolTable rtol_table = default_rtol_table>
        bool parse_expr       (const TokenArray& tarr, AstExpr& ast_expr, int indent_level);
        bool parse_module     (const TokenArray& tarr, AstModule& ast_module);
    }
}

#endif
