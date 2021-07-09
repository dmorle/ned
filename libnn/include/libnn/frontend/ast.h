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
        class AstDef;
        class Module
        {
        public:
            std::vector<ModImp> imps;
            std::vector<AstDef> defs;

            Module(const TokenArray& tarr);
            ~Module();

            Obj* eval(const std::string& entry_point, EvalCtx& ctx);
        };

        class Ast
        {
        public:
            virtual Obj* eval(EvalCtx& ctx, Module& mod) = 0;
        };

        // root node
        class AstDef :
            public Ast
        {
        public:
            AstDef(const TokenArray& tarr, int indent_level);
            ~AstDef();

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstFor :
            public Ast
        {
        public:
            AstFor(const TokenArray& tarr, int indent_level);
            ~AstFor();

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstWhile :
            public Ast
        {
        public:
            AstWhile(const TokenArray& tarr, int indent_level);
            ~AstWhile();

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstIf :
            public Ast
        {
        public:
            AstIf(const TokenArray& tarr, int indent_level);
            ~AstIf();

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstSequence :
            public Ast
        {
        public:
            AstSequence(const TokenArray& tarr, int indent_level);
            ~AstSequence();

            virtual Obj* eval(EvalCtx& ctx, Module& mod);
        };

        class AstIAdd :
            public Ast
        {
        public:
            AstIAdd(const TokenArray& tarr);
        };

        class AstDecl :
            public Ast
        {
        public:
            AstDecl(const TokenArray& tarr);
        };

        class AstIdn :
            public Ast
        {
        public:
            AstIdn(const TokenArray& tarr);
        };

        class AstTuple :
            public Ast
        {
        public:
            AstTuple(const TokenArray& tarr);
        };

        class AstStr :
            public Ast
        {
        public:
            AstStr(const TokenArray& tarr);
        };

        class AstFloat :
            public Ast
        {
        public:
            AstFloat(const TokenArray& tarr);
        };

        class AstInt :
            public Ast
        {
        public:
            AstInt(const TokenArray& tarr);
        };

        class AstBool :
            public Ast
        {
        public:
            AstBool(const TokenArray& tarr);
        };
    }
}

#endif
