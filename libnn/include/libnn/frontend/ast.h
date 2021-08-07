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
        class AstModule;

        class AstBlock
        {
        public:
            std::string file_name;
            uint32_t line_num;
            uint32_t col_num;

            virtual ~AstBlock() = 0;

            virtual Obj* eval(EvalCtx& ctx) const = 0;
        };

        class AstCargsDecl
        {
        public:
            virtual ~AstCargsDecl() = 0;

            // does not evaluate anything or add new elements to the scope
            // potentially 
            virtual void match_args(Scope& scope, std::vector<Obj*>& cargs) const = 0;
        };

        class AstExpr :
            public AstBlock
        {
        public:
            virtual ~AstExpr() = 0;
            
            virtual void append_cargs(EvalCtx&, std::vector<Obj*>&) const;
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

            virtual Obj* eval(EvalCtx& ctx) const;
        };

        class AstInt :
            public AstExpr
        {
            int64_t val;

        public:
            AstInt(const Token* ptk);

            virtual Obj* eval(EvalCtx& ctx) const;
        };

        class AstFloat :
            public AstExpr
        {
            double val;

        public:
            AstFloat(const Token* ptk);

            virtual Obj* eval(EvalCtx& ctx) const;
        };

        class AstStr :
            public AstExpr
        {
            std::string val;

        public:
            AstStr(const Token* ptk);

            virtual Obj* eval(EvalCtx& ctx) const;
        };

        class AstIdn :
            public AstExpr
        {
            std::string idn;
            
        public:
            AstIdn(const Token* ptk);

            virtual Obj* eval(EvalCtx& ctx) const;
        };

        class AstTuple :
            public AstExpr
        {
            std::vector<AstExpr*> elems;

        public:
            AstTuple(const TokenArray& tarr);
            virtual ~AstTuple();

            virtual Obj* eval(EvalCtx& ctx) const;
        };

        class AstCall :
            public AstExpr
        {
            AstExpr* pleft;
            std::vector<AstExpr*> args;

        public:
            AstCall(AstExpr* pleft, const TokenArray& tarr);
            virtual Obj* eval(EvalCtx& ctx) const;
        };

        class AstCargs :
            public AstExpr
        {
            AstExpr* pleft;
            std::vector<AstExpr*> args;

        public:
            AstCargs(AstExpr* pleft, const TokenArray& tarr);

            virtual Obj* eval(EvalCtx& ctx) const;
        };

        class AstIdx :
            public AstExpr
        {
            AstExpr* pleft;
            std::vector<std::vector<AstExpr*>> indicies;

        public:
            AstIdx(AstExpr* pleft, const TokenArray& tarr);
            void parseSlice(const TokenArray& tarr);

            virtual Obj* eval(EvalCtx& ctx) const;
        };

        class AstDot :
            public AstExpr
        {
            AstExpr* pleft;
            std::string member;

        public:
            AstDot(AstExpr* pleft, const Token* ptk);

            virtual Obj* eval(EvalCtx& ctx) const;
        };

        class AstNeg :
            public AstExpr
        {
            AstExpr* pexpr;

        public:
            AstNeg(const TokenArray& tarr);

            virtual Obj* eval(EvalCtx& ctx) const;
        };

        class AstPack :
            public AstExpr
        {
            AstExpr* pexpr;

        public:
            AstPack(const TokenArray& tarr);

            virtual Obj* eval(EvalCtx& ctx) const;
            virtual void append_cargs(EvalCtx&, std::vector<Obj*>&);
        };

        class AstAdd :
            public AstBinOp
        {
        public:
            AstAdd(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx) const;
        };
        
        class AstSub :
            public AstBinOp
        {
        public:
            AstSub(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx) const;
        };
        
        class AstMul :
            public AstBinOp
        {
        public:
            AstMul(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx) const;
        };
        
        class AstDiv :
            public AstBinOp
        {
        public:
            AstDiv(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx) const;
        };

        class AstEq :
            public AstBinOp
        {
        public:
            AstEq(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx) const;
        };

        class AstNe :
            public AstBinOp
        {
        public:
            AstNe(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx) const;
        };

        class AstGe :
            public AstBinOp
        {
        public:
            AstGe(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx) const;
        };

        class AstLe :
            public AstBinOp
        {
        public:
            AstLe(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx) const;
        };

        class AstGt :
            public AstBinOp
        {
        public:
            AstGt(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx) const;
        };

        class AstLt :
            public AstBinOp
        {
        public:
            AstLt(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx) const;
        };

        class AstAnd :
            public AstBinOp
        {
        public:
            AstAnd(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx) const;
        };

        class AstOr :
            public AstBinOp
        {
        public:
            AstOr(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx) const;
        };

        class AstIAdd :
            public AstBinOp
        {
        public:
            AstIAdd(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx) const;
        };

        class AstISub :
            public AstBinOp
        {
        public:
            AstISub(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx) const;
        };

        class AstIMul :
            public AstBinOp
        {
        public:
            AstIMul(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx) const;
        };

        class AstIDiv :
            public AstBinOp
        {
        public:
            AstIDiv(const TokenArray& left, const TokenArray& right);

            virtual Obj* eval(EvalCtx& ctx) const;
        };

        class AstAssign :
            public AstBinOp
        {
            bool decl_assign;

        public:
            AstAssign(const TokenArray& left, const TokenArray& right);
            virtual ~AstAssign();

            virtual Obj* eval(EvalCtx& ctx) const;
        };

        // end of expression nodes
        // block nodes

        // simple variable delarations found in sequences
        class AstDecl :
            public AstExpr,
            public AstCargsDecl
        {
            friend class AstModule;

            bool is_static;
            std::string var_name;
            std::string type_name;
            std::vector<AstExpr*> cargs;

        public:
            AstDecl();
            AstDecl(const TokenArray& tarr);
            virtual ~AstDecl();

            //  constructs a new object with the given cargs
            //  adds the new object's name to the scope
            //  if state is CALL:
            //      assert cargs are fully initialized
            //  elif state is DEFSEQ and type is tensor: 
            //      add the tensor as a graph input edge
            //
            virtual Obj* eval(EvalCtx& ctx) const;  // returns null
            virtual void append_cargs(EvalCtx&, std::vector<Obj*>&) const;
            virtual void match_args(Scope& scope, std::vector<Obj*>& cargs) const;
        };

        class AstSeq
        {
            uint32_t line_num;
            uint32_t col_num;

            std::vector<AstBlock*> blocks;

        public:
            AstSeq(const TokenArray& tarr, int indent_level);
            ~AstSeq();

            virtual Obj* eval(EvalCtx& ctx) const;
        };

        class AstIf :
            public AstBlock
        {
            AstExpr* pcond;
            AstSeq seq;

        public:
            AstIf(const TokenArray& if_sig, const TokenArray& if_seq, int indent_level);
            virtual ~AstIf();

            virtual Obj* eval(EvalCtx& ctx) const;
        };

        class AstWhile :
            public AstBlock
        {
            AstExpr* pcond;
            AstSeq seq;

        public:
            AstWhile(const TokenArray& while_sig, const TokenArray& whlie_seq, int indent_level);
            virtual ~AstWhile();

            virtual Obj* eval(EvalCtx& ctx) const;
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

            virtual Obj* eval(EvalCtx& ctx) const;
        };

        class AstCargsTuple :
            public AstCargsDecl
        {
            std::vector<AstCargsDecl*> elems;

        public:
            AstCargsTuple(const TokenArray& tarr);
            virtual ~AstCargsTuple();

            virtual void match_args(Scope& scope, std::vector<Obj*>& cargs) const;
        };

        // root node
        class AstDef
        {
            friend class AstModule;

            uint32_t line_num;
            uint32_t col_num;

            AstSeq block;
            std::string name;

            std::vector<AstCargsDecl*> cargs;
            std::vector<AstDecl> vargs;

        public:
            AstDef(const TokenArray& def_sig, const TokenArray& def_block);

            void eval(EvalCtx& ctx) const;
        };

        class AstIntr
        {
        public:
            void eval(EvalCtx& ctx) const;
        };

        class AstFn
        {
        public:
            void eval(EvalCtx& ctx) const;
        };

        class AstModImp
        {
            friend class AstModule;

            uint32_t line_num;
            uint32_t col_num;

            std::vector<std::string> imp;

        public:
            AstModImp(const TokenArray& tarr);

            void eval(EvalCtx& ctx) const;
        };

        class AstModule
        {
            std::vector<AstModImp> imps;
            std::vector<AstDef> defs;
            std::vector<AstFn> fns;
            std::vector<AstIntr> intrs;

        public:
            AstModule(const TokenArray& tarr);

            EvalCtx* eval(const std::string& entry_point, std::vector<Obj*>& cargs);
        };
    }
}

#endif
