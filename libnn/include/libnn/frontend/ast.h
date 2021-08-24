#ifndef NN_PARSER_H
#define NN_PARSER_H

#include <string>
#include <vector>
#include <tuple>
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

            virtual std::shared_ptr<Obj> eval(EvalCtx& ctx) const = 0;
        };

        class AstExpr :
            public AstBlock
        {
        public:
            virtual ~AstExpr() = 0;
            
            virtual void append_vec(EvalCtx&, std::vector<std::shared_ptr<Obj>>&) const;
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

            virtual std::shared_ptr<Obj> eval(EvalCtx& ctx) const;
        };

        class AstInt :
            public AstExpr
        {
            int64_t val;

        public:
            AstInt(const Token* ptk);

            virtual std::shared_ptr<Obj> eval(EvalCtx& ctx) const;
        };

        class AstFloat :
            public AstExpr
        {
            double val;

        public:
            AstFloat(const Token* ptk);

            virtual std::shared_ptr<Obj> eval(EvalCtx& ctx) const;
        };

        class AstStr :
            public AstExpr
        {
            std::string val;

        public:
            AstStr(const Token* ptk);

            virtual std::shared_ptr<Obj> eval(EvalCtx& ctx) const;
        };

        class AstIdn :
            public AstExpr
        {
            std::string idn;
            
        public:
            AstIdn();
            AstIdn(const Token* ptk);

            virtual std::shared_ptr<Obj> eval(EvalCtx& ctx) const;
        };

        class AstTuple :
            public AstExpr
        {
            std::vector<AstExpr*> elems;

        public:
            AstTuple(const TokenArray& tarr);
            virtual ~AstTuple();

            virtual std::shared_ptr<Obj> eval(EvalCtx& ctx) const;
        };

        class AstCall :
            public AstExpr
        {
            AstExpr* pleft;
            std::vector<AstExpr*> args;

        public:
            AstCall(AstExpr* pleft, const TokenArray& tarr);
            virtual std::shared_ptr<Obj> eval(EvalCtx& ctx) const;
        };

        class AstCargs :
            public AstExpr
        {
            AstExpr* pleft;
            std::vector<AstExpr*> args;

        public:
            AstCargs(AstExpr* pleft, const TokenArray& tarr);

            virtual std::shared_ptr<Obj> eval(EvalCtx& ctx) const;
        };

        class AstIdx :
            public AstExpr
        {
            AstExpr* pleft;
            AstExpr* pidx;
            //std::vector<std::vector<AstExpr*>> indicies;

        public:
            AstIdx(AstExpr* pleft, const TokenArray& tarr);
            //void parseSlice(const TokenArray& tarr);

            virtual std::shared_ptr<Obj> eval(EvalCtx& ctx) const;
        };

        class AstDot :
            public AstExpr
        {
            AstExpr* pleft;
            std::string member;

        public:
            AstDot(AstExpr* pleft, const Token* ptk);

            virtual std::shared_ptr<Obj> eval(EvalCtx& ctx) const;
        };

        class AstNeg :
            public AstExpr
        {
            AstExpr* pexpr;

        public:
            AstNeg(const TokenArray& tarr);

            virtual std::shared_ptr<Obj> eval(EvalCtx& ctx) const;
        };

        class AstPack :
            public AstExpr
        {
            AstExpr* pexpr;

        public:
            AstPack(const TokenArray& tarr);

            virtual std::shared_ptr<Obj> eval(EvalCtx& ctx) const;
            virtual void append_vec(EvalCtx&, std::vector<std::shared_ptr<Obj>>&) const;
        };

        class AstAdd :
            public AstBinOp
        {
        public:
            AstAdd(const TokenArray& left, const TokenArray& right);

            virtual std::shared_ptr<Obj> eval(EvalCtx& ctx) const;
        };
        
        class AstSub :
            public AstBinOp
        {
        public:
            AstSub(const TokenArray& left, const TokenArray& right);

            virtual std::shared_ptr<Obj> eval(EvalCtx& ctx) const;
        };
        
        class AstMul :
            public AstBinOp
        {
        public:
            AstMul(const TokenArray& left, const TokenArray& right);

            virtual std::shared_ptr<Obj> eval(EvalCtx& ctx) const;
        };
        
        class AstDiv :
            public AstBinOp
        {
        public:
            AstDiv(const TokenArray& left, const TokenArray& right);

            virtual std::shared_ptr<Obj> eval(EvalCtx& ctx) const;
        };

        class AstEq :
            public AstBinOp
        {
        public:
            AstEq(const TokenArray& left, const TokenArray& right);

            virtual std::shared_ptr<Obj> eval(EvalCtx& ctx) const;
        };

        class AstNe :
            public AstBinOp
        {
        public:
            AstNe(const TokenArray& left, const TokenArray& right);

            virtual std::shared_ptr<Obj> eval(EvalCtx& ctx) const;
        };

        class AstGe :
            public AstBinOp
        {
        public:
            AstGe(const TokenArray& left, const TokenArray& right);

            virtual std::shared_ptr<Obj> eval(EvalCtx& ctx) const;
        };

        class AstLe :
            public AstBinOp
        {
        public:
            AstLe(const TokenArray& left, const TokenArray& right);

            virtual std::shared_ptr<Obj> eval(EvalCtx& ctx) const;
        };

        class AstGt :
            public AstBinOp
        {
        public:
            AstGt(const TokenArray& left, const TokenArray& right);

            virtual std::shared_ptr<Obj> eval(EvalCtx& ctx) const;
        };

        class AstLt :
            public AstBinOp
        {
        public:
            AstLt(const TokenArray& left, const TokenArray& right);

            virtual std::shared_ptr<Obj> eval(EvalCtx& ctx) const;
        };

        class AstAnd :
            public AstBinOp
        {
        public:
            AstAnd(const TokenArray& left, const TokenArray& right);

            virtual std::shared_ptr<Obj> eval(EvalCtx& ctx) const;
        };

        class AstOr :
            public AstBinOp
        {
        public:
            AstOr(const TokenArray& left, const TokenArray& right);

            virtual std::shared_ptr<Obj> eval(EvalCtx& ctx) const;
        };

        class AstIAdd :
            public AstBinOp
        {
        public:
            AstIAdd(const TokenArray& left, const TokenArray& right);

            virtual std::shared_ptr<Obj> eval(EvalCtx& ctx) const;
        };

        class AstISub :
            public AstBinOp
        {
        public:
            AstISub(const TokenArray& left, const TokenArray& right);

            virtual std::shared_ptr<Obj> eval(EvalCtx& ctx) const;
        };

        class AstIMul :
            public AstBinOp
        {
        public:
            AstIMul(const TokenArray& left, const TokenArray& right);

            virtual std::shared_ptr<Obj> eval(EvalCtx& ctx) const;
        };

        class AstIDiv :
            public AstBinOp
        {
        public:
            AstIDiv(const TokenArray& left, const TokenArray& right);

            virtual std::shared_ptr<Obj> eval(EvalCtx& ctx) const;
        };

        class AstAssign :
            public AstBinOp
        {
            bool decl_assign;

        public:
            AstAssign(const TokenArray& left, const TokenArray& right);
            virtual ~AstAssign();

            virtual std::shared_ptr<Obj> eval(EvalCtx& ctx) const;
        };

        // end of expression nodes
        // block nodes

        // simple variable delarations found in sequences
        class AstDecl :
            public AstExpr
        {
            friend class AstModule;

            bool is_static;
            std::string var_name;
            AstIdn type_idn;
            std::vector<AstExpr*> cargs;
            bool has_cargs;

        public:
            AstDecl();
            AstDecl(const TokenArray& tarr);
            virtual ~AstDecl();

            //  constructs a new object with the given cargs
            //  adds the new object's name to the scope
            //  if state is DEFSEQ and type is tensor: 
            //      add the tensor as a graph input edge
            //
            virtual std::shared_ptr<Obj> eval(EvalCtx& ctx) const override;  // returns null
        };

        class AstReturn :
            public AstBlock
        {
            AstExpr* ret;

        public:
            AstReturn(const TokenArray& tarr);
            virtual ~AstReturn();

            virtual std::shared_ptr<Obj> eval(EvalCtx& ctx) const;
        };

        class AstRaise :
            public AstBlock
        {
            AstExpr* val;

        public:
            AstRaise(const TokenArray& tarr);
            virtual ~AstRaise();

            virtual std::shared_ptr<Obj> eval(EvalCtx& ctx) const;
        };

        class AstSeq
        {
            uint32_t line_num;
            uint32_t col_num;

            std::vector<AstBlock*> blocks;

        public:
            AstSeq(const TokenArray& tarr, int indent_level);
            ~AstSeq();

            virtual std::shared_ptr<Obj> eval(EvalCtx& ctx) const;
        };

        class AstIf :
            public AstBlock
        {
            AstExpr* pcond;
            AstSeq seq;

        public:
            AstIf(const TokenArray& if_sig, const TokenArray& if_seq, int indent_level);
            virtual ~AstIf();

            virtual std::shared_ptr<Obj> eval(EvalCtx& ctx) const;
        };

        class AstWhile :
            public AstBlock
        {
            AstExpr* pcond;
            AstSeq seq;

        public:
            AstWhile(const TokenArray& while_sig, const TokenArray& whlie_seq, int indent_level);
            virtual ~AstWhile();

            virtual std::shared_ptr<Obj> eval(EvalCtx& ctx) const;
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

            virtual std::shared_ptr<Obj> eval(EvalCtx& ctx) const;
        };

        class AstCargSig
        {
        public:
            virtual ~AstCargSig() = 0;

            virtual std::vector<std::shared_ptr<Obj>>::iterator match_args(
                EvalCtx& ctx,
                std::vector<std::shared_ptr<Obj>>::iterator start,
                std::vector<std::shared_ptr<Obj>>::iterator end) const = 0;
        };

        class AstCargDecl :
            public AstCargSig
        {
            uint32_t line_num;
            uint32_t col_num;

            bool is_packed;
            std::string var_name;
            AstIdn type_idn;
            std::vector<AstExpr*> cargs;
            bool has_cargs;

        public:
            AstCargDecl(const TokenArray& tarr);
            virtual ~AstCargDecl();

            virtual std::vector<std::shared_ptr<Obj>>::iterator match_args(
                EvalCtx& ctx,
                std::vector<std::shared_ptr<Obj>>::iterator start,
                std::vector<std::shared_ptr<Obj>>::iterator end) const override;
        };

        class AstCargTuple :
            public AstCargSig
        {
            std::vector<AstCargSig*> elems;

        public:
            AstCargTuple(const TokenArray& tarr);
            virtual ~AstCargTuple();

            virtual std::vector<std::shared_ptr<Obj>>::iterator match_args(
                EvalCtx& ctx,
                std::vector<std::shared_ptr<Obj>>::iterator start,
                std::vector<std::shared_ptr<Obj>>::iterator end) const override;
        };

        class AstArgSig
        {
        public:
            virtual ~AstArgSig();

            using Iter = std::vector<std::shared_ptr<Obj>>::iterator;
            virtual Iter carg_deduction(EvalCtx& ctx, const Iter& start, const Iter& end) const = 0;
        };

        // Immediate arg carg parameter (any expression)
        class AstArgImm :
            public AstArgSig
        {
            AstExpr* pimm;

        public:
            AstArgImm(const TokenArray& tarr);
            virtual ~AstArgImm();

            virtual Iter carg_deduction(EvalCtx& ctx, const Iter& start, const Iter& end) const override;
        };

        class AstArgVar :
            public AstArgSig
        {
            bool is_packed;
            std::string var_name;

        public:
            AstArgVar(const TokenArray& tarr);

            virtual Iter carg_deduction(EvalCtx& ctx, const Iter& start, const Iter& end) const override;
        };

        class AstArgDecl :
            public AstArgSig
        {
            std::string type_name;
            std::vector<AstArgSig*> cargs;
            bool has_cargs;
            
        public:
            AstArgDecl(const TokenArray& tarr);
            virtual ~AstArgDecl();

            virtual Iter carg_deduction(EvalCtx& ctx, const Iter& start, const Iter& end) const override;
            std::shared_ptr<Obj> auto_gen(EvalCtx& ctx, const std::string& name) const;
        };

        // root node
        class AstDef
        {
            friend class AstModule;

            uint32_t line_num;
            uint32_t col_num;

            AstSeq block;
            std::string name;

            AstCargTuple* cargs;
            std::vector<std::pair<AstArgDecl, std::string>> vargs;

        public:
            AstDef(const TokenArray& def_sig, const TokenArray& def_block);
            ~AstDef();

            void eval(EvalCtx& ctx) const;
            void apply_cargs(EvalCtx& ctx, std::vector<std::shared_ptr<Obj>>& cargs) const;
            void carg_deduction(EvalCtx& ctx, std::vector<std::shared_ptr<Obj>>& args) const;
            
            const std::string& get_name() const;
            const AstSeq& get_body() const;
        };

        class AstIntr
        {
            uint32_t line_num;
            uint32_t col_num;
            
            AstSeq block;
            std::string name;

            AstCargTuple* cargs;
            std::vector<std::pair<AstArgDecl, std::string>> vargs;

        public:
            AstIntr(const TokenArray& intr_sig, const TokenArray& intr_block);
            ~AstIntr();

            void eval(EvalCtx& ctx) const;
            void apply_cargs(EvalCtx& ctx, std::vector<std::shared_ptr<Obj>>& cargs) const;
            void carg_deduction(EvalCtx& ctx, std::vector<std::shared_ptr<Obj>>& args) const;

            const std::string& get_name() const;
            const AstSeq& get_body() const;
        };

        class AstFn
        {
            uint32_t line_num;
            uint32_t col_num;

            AstSeq block;
            std::string name;

            // fn args behave the same as def/intr cargs
            AstCargTuple* args;

        public:
            AstFn(const TokenArray& fn_sig, const TokenArray& fn_block);

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

            EvalCtx* eval(const std::string& entry_point, std::vector<std::shared_ptr<Obj>>& cargs);
        };
    }
}

#endif
