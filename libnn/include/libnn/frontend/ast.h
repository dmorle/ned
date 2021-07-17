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
            virtual Obj* eval(EvalCtx& ctx, Module& mod) = 0;
        };

        class AstExpr :
            public AstBlock
        {};

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
            ~AstCall();

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        AstExpr* parseExpr(const TokenArray& tarr);

        // end of expression nodes
        // block nodes

        class AstDecl :
            public AstBlock
        {
        public:
            AstDecl();
            AstDecl(const TokenArray& tarr);

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstIf :
            public AstBlock
        {
        public:
            AstIf(const TokenArray& if_sig, const TokenArray& if_seq, int indent_level);
            ~AstIf();

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstWhile :
            public AstBlock
        {
        public:
            AstWhile(const TokenArray& while_sig, const TokenArray& whlie_seq, int indent_level);
            ~AstWhile();

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstFor :
            public AstBlock
        {
            AstDecl it;
            AstSeq seq;
            AstExpr* pexpr;

        public:
            AstFor(const TokenArray& for_sig, const TokenArray& for_seq, int indent_level);
            ~AstFor();

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

        // root node
        class AstDef
        {
            AstSeq block;
            std::string name;
            std::vector<AstDecl> constargs;
            std::vector<AstDecl> varargs;

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
