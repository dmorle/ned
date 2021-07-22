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
        public:
            AstBool(const TokenArray& tarr);
        };

        class AstInt :
            public AstExpr
        {
        public:
            AstInt(const TokenArray& tarr);
        };

        class AstFloat :
            public AstExpr
        {
        public:
            AstFloat(const TokenArray& tarr);
        };

        class AstStr :
            public AstExpr
        {
        public:
            AstStr(const TokenArray& tarr);
        };

        class AstTuple :
            public AstExpr
        {
        public:
            AstTuple(const TokenArray& tarr);
            virtual ~AstTuple();

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstIdn :
            public AstExpr
        {
        public:
            AstIdn(const TokenArray& tarr);
        };

        class AstIAdd :
            public AstExpr
        {
        public:
            AstIAdd(const TokenArray& tarr);
        };

        class AstCall :
            public AstExpr
        {
        public:
            AstCall(const TokenArray& tarr, int indent_level);
            virtual ~AstCall();

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        AstExpr* parseExpr(const TokenArray& tarr);

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
