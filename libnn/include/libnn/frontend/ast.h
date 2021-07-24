#ifndef NN_PARSER_H
#define NN_PARSER_H

#include <string>
#include <vector>
#include <unordered_map>

#include <libnn/core/graph.h>
#include <libnn/frontend/lexer.h>

namespace nn
{
    namespace frontend
    {
        class Obj;
        using EvalCtx = std::unordered_map<std::string, Obj*>;

        class ModImp;
        class AstSeq;
        class AstDef;
        class Module;

        class AstBlock
        {
        public:
            virtual ~AstBlock() = 0;

            virtual Obj* eval(EvalCtx& ctx, Module& mod) = 0;
        };

        class AstExpr :
            public AstBlock
        {
        public:
            virtual ~AstExpr() = 0;
        };

        class AstBool :
            public AstExpr
        {
            bool val;

        public:
            AstBool(bool val);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstInt :
            public AstExpr
        {
            int64_t val;

        public:
            AstInt(int64_t val);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstFloat :
            public AstExpr
        {
            double val;

        public:
            AstFloat(double val);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstStr :
            public AstExpr
        {
            std::string val;

        public:
            AstStr(const std::string& val);

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

        class AstIdn :
            public AstExpr
        {
            std::string idn;
            
        public:
            AstIdn(const std::string& idn);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstCall :
            public AstExpr
        {
        public:
            AstCall(AstExpr* pleft, const TokenArray& tarr);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstCargs :
            public AstExpr
        {
        public:
            AstCargs(AstExpr* pleft, const TokenArray& tarr);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstSlice :
            public AstExpr
        {
        public:
            AstSlice(AstExpr* pleft, const TokenArray& tarr);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstDot :
            public AstExpr
        {
            AstExpr* pleft;
            std::string member;

        public:
            AstDot(AstExpr* pleft, const std::string& member);

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
            public AstExpr
        {
            AstExpr* pleft;
            AstExpr* pright;

        public:
            AstAdd(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };
        
        class AstSub :
            public AstExpr
        {
            AstExpr* pleft;
            AstExpr* pright;

        public:
            AstSub(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };
        
        class AstMul :
            public AstExpr
        {
            AstExpr* pleft;
            AstExpr* pright;

        public:
            AstMul(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };
        
        class AstDiv :
            public AstExpr
        {
            AstExpr* pleft;
            AstExpr* pright;

        public:
            AstDiv(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstEq :
            public AstExpr
        {
            AstExpr* pleft;
            AstExpr* pright;

        public:
            AstEq(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstNe :
            public AstExpr
        {
            AstExpr* pleft;
            AstExpr* pright;

        public:
            AstNe(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstGe :
            public AstExpr
        {
            AstExpr* pleft;
            AstExpr* pright;

        public:
            AstGe(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstLe :
            public AstExpr
        {
            AstExpr* pleft;
            AstExpr* pright;

        public:
            AstLe(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstGt :
            public AstExpr
        {
            AstExpr* pleft;
            AstExpr* pright;

        public:
            AstGt(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstLt :
            public AstExpr
        {
            AstExpr* pleft;
            AstExpr* pright;

        public:
            AstLt(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstAnd :
            public AstExpr
        {
            AstExpr* pleft;
            AstExpr* pright;

        public:
            AstAnd(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstOr :
            public AstExpr
        {
            AstExpr* pleft;
            AstExpr* pright;

        public:
            AstOr(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstIAdd :
            public AstExpr
        {
            AstExpr* pleft;
            AstExpr* pright;

        public:
            AstIAdd(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstISub :
            public AstExpr
        {
            AstExpr* pleft;
            AstExpr* pright;

        public:
            AstISub(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstIMul :
            public AstExpr
        {
            AstExpr* pleft;
            AstExpr* pright;

        public:
            AstIMul(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstIDiv :
            public AstExpr
        {
            AstExpr* pleft;
            AstExpr* pright;

        public:
            AstIDiv(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstAssign :
            public AstExpr
        {
            AstExpr* pleft;
            AstExpr* pright;

        public:
            AstAssign(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        // end of expression nodes
        // block nodes

        // simple variable delarations found in sequences
        class AstDecl :
            public AstBlock
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

        class AstSeq
        {
            std::vector<AstBlock*> blocks;

        public:
            AstSeq(const TokenArray& tarr, int indent_level);
            ~AstSeq();

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
            AstSeq block;
            std::string name;
            std::vector<AstDefCarg*> constargs;
            std::vector<AstDefArgSingle> varargs;

        public:
            AstDef(const TokenArray& def_sig, const TokenArray& def_block);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class Module
        {
            std::vector<ModImp> imps;
            std::vector<AstDef> defs;

        public:
            Module(const TokenArray& tarr);

            Obj* eval(const std::string& entry_point, EvalCtx& ctx);
        };
    }
}

#endif
