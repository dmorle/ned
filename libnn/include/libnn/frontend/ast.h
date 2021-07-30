#ifndef NN_PARSER_H
#define NN_PARSER_H

#include <string>
#include <vector>
#include <stack>
#include <unordered_map>

#include <libnn/core/graph.h>
#include <libnn/frontend/eval.h>
#include <libnn/frontend/lexer.h>

namespace nn
{
    namespace impl
    {
        class AstSeq;
        class AstDef;
        class AstIntr;
        class Module;

        class AstBlock
        {
        public:
            std::string file_name;
            uint32_t line_num;
            uint32_t col_num;

            virtual ~AstBlock() = 0;

            virtual Obj* eval(EvalCtx& ctx, Module& mod) = 0;
        };

        class AstExpr :
            public AstBlock
        {
        public:
            virtual ~AstExpr() = 0;
        };

        class AstBinOp :
            public AstExpr
        {
        protected:
            AstExpr* pleft;
            AstExpr* pright;

        public:
            virtual ~AstBinOp();
        };

        class AstBool :
            public AstExpr
        {
            bool val;

        public:
            AstBool(const Token* ptk, bool val);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstInt :
            public AstExpr
        {
            int64_t val;

        public:
            AstInt(const Token* ptk);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstFloat :
            public AstExpr
        {
            double val;

        public:
            AstFloat(const Token* ptk);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstStr :
            public AstExpr
        {
            std::string val;

        public:
            AstStr(const Token* ptk);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstIdn :
            public AstExpr
        {
            std::string idn;
            
        public:
            AstIdn(const Token* ptk);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstTuple :
            public AstExpr
        {
            std::vector<AstExpr*> elems;

        public:
            AstTuple(const TokenArray& tarr);
            virtual ~AstTuple();

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstCall :
            public AstExpr
        {
            AstExpr* pleft;
            std::vector<AstExpr*> args;

        public:
            AstCall(AstExpr* pleft, const TokenArray& tarr);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstCargs :
            public AstExpr
        {
            AstExpr* pleft;
            std::vector<AstExpr*> args;

        public:
            AstCargs(AstExpr* pleft, const TokenArray& tarr);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstIdx :
            public AstExpr
        {
            AstExpr* pleft;
            std::vector<std::vector<AstExpr*>> indicies;

        public:
            AstIdx(AstExpr* pleft, const TokenArray& tarr);
            void parseSlice(const TokenArray& tarr);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstDot :
            public AstExpr
        {
            AstExpr* pleft;
            std::string member;

        public:
            AstDot(AstExpr* pleft, const Token* ptk);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstNeg :
            public AstExpr
        {
            AstExpr* pexpr;

        public:
            AstNeg(const TokenArray& tarr);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstAdd :
            public AstBinOp
        {
        public:
            AstAdd(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };
        
        class AstSub :
            public AstBinOp
        {
        public:
            AstSub(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };
        
        class AstMul :
            public AstBinOp
        {
        public:
            AstMul(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };
        
        class AstDiv :
            public AstBinOp
        {
        public:
            AstDiv(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstEq :
            public AstBinOp
        {
        public:
            AstEq(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstNe :
            public AstBinOp
        {
        public:
            AstNe(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstGe :
            public AstBinOp
        {
        public:
            AstGe(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstLe :
            public AstBinOp
        {
        public:
            AstLe(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstGt :
            public AstBinOp
        {
        public:
            AstGt(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstLt :
            public AstBinOp
        {
        public:
            AstLt(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstAnd :
            public AstBinOp
        {
        public:
            AstAnd(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstOr :
            public AstBinOp
        {
        public:
            AstOr(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstIAdd :
            public AstBinOp
        {
        public:
            AstIAdd(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstISub :
            public AstBinOp
        {
        public:
            AstISub(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstIMul :
            public AstBinOp
        {
        public:
            AstIMul(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstIDiv :
            public AstBinOp
        {
        public:
            AstIDiv(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstAssign :
            public AstBinOp
        {
            bool decl_assign;

        public:
            AstAssign(const TokenArray& left, const TokenArray& right);
            virtual ~AstAssign();

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        // end of expression nodes
        // block nodes

        // simple variable delarations found in sequences
        class AstDecl :
            public AstExpr
        {
            std::string var_name;
            std::string type_name;
            std::vector<AstExpr*> constargs;

        public:
            AstDecl();
            AstDecl(const TokenArray& tarr);
            virtual ~AstDecl();

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstSeq
        {
            uint32_t line_num;
            uint32_t col_num;

            std::vector<AstBlock*> blocks;

        public:
            AstSeq(const TokenArray& tarr, int indent_level);
            ~AstSeq();

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstIf :
            public AstBlock
        {
            AstExpr* pcond;
            AstSeq seq;

        public:
            AstIf(const TokenArray& if_sig, const TokenArray& if_seq, int indent_level);
            virtual ~AstIf();

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstWhile :
            public AstBlock
        {
            AstExpr* pcond;
            AstSeq seq;

        public:
            AstWhile(const TokenArray& while_sig, const TokenArray& whlie_seq, int indent_level);
            virtual ~AstWhile();

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstFor :
            public AstBlock
        {
            AstDecl it;
            AstExpr* pexpr;
            AstSeq seq;

        public:
            AstFor(const TokenArray& for_sig, const TokenArray& for_seq, int indent_level);
            virtual ~AstFor();

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        // constargs can have tuples, varargs can't
        class AstDefCarg
        {
        public:
            virtual ~AstDefCarg() = 0;
        };

        // basically AstDecl but with packed parameters
        class AstDefArgSingle :
            public AstDefCarg
        {
        public:
            bool packed;
            std::string var_name;
            std::string type_name;
            std::vector<AstExpr*> constargs;

            AstDefArgSingle(const TokenArray& tarr);
            virtual ~AstDefArgSingle();
        };

        // Tuple of AstDefCarg s
        class AstDefCargTuple :
            public AstDefCarg
        {
        public:
            std::vector<AstDefCarg*> elems;

            AstDefCargTuple(const TokenArray& tarr);
            virtual ~AstDefCargTuple();
        };

        // root node
        class AstDef
        {
            uint32_t line_num;
            uint32_t col_num;

            AstSeq block;
            std::string name;
            std::vector<AstDefCarg*> constargs;
            std::vector<AstDefArgSingle> varargs;

        public:
            AstDef(const TokenArray& def_sig, const TokenArray& def_block);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstIntr
        {
            // TODO: figure out instrinsics
        };

        class AstModImp :
            AstBlock
        {
            std::vector<std::string> imp;

        public:
            AstModImp(const TokenArray& tarr);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class Module
        {
            std::vector<AstModImp> imps;
            std::vector<AstDef> defs;

        public:
            Module(const TokenArray& tarr);

            Obj* eval(const std::string& entry_point, EvalCtx& ctx);
        };
    }
}

#endif
