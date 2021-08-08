#include <libnn/frontend/eval.h>
#include <libnn/frontend/obj.h>
#include <libnn/frontend/ast.h>

#include <cassert>

namespace nn
{
    namespace impl
    {
        // Definitions for eval.h

        void AstExpr::append_vec(EvalCtx& ctx, std::vector<std::shared_ptr<Obj>>& cargs) const
        {
            cargs.push_back(eval(ctx));
        }

        void AstPack::append_vec(EvalCtx& ctx, std::vector<std::shared_ptr<Obj>>& cargs)
        {
            std::vector<std::shared_ptr<Obj>> eval_result = std::move(eval(ctx)->iter(ctx));
            cargs.insert(
                cargs.end(),
                std::make_move_iterator(eval_result.begin()),
                std::make_move_iterator(eval_result.end())
            );
        }

        GenerationError::GenerationError(const std::string& errmsg) :
            errmsg(errmsg)
        {}

        void GenerationError::tb_touch(AstBlock* pblock)
        {
            traceback.push_back({ pblock->file_name, pblock->line_num, pblock->col_num });
        }

        void GenerationError::tb_touch(const std::string& file_name, uint32_t line_num, uint32_t col_num)
        {
            traceback.push_back({ file_name, line_num, col_num });
        }

        EvalCtx::EvalCtx()
        {
            model_params = {};
            pgraph = new Graph();
            pscope = nullptr;

            state = EvalState::CALL;
        }

        // AST node evaluation

        static std::unordered_map<std::string, int> varcounts = {};

        std::shared_ptr<Obj> AstBool::eval(EvalCtx& ctx) const
        {
            return create_obj<ObjType::BOOL>(val);
        }

        std::shared_ptr<Obj> AstInt::eval(EvalCtx& ctx) const
        {
            return create_obj<ObjType::INT>(val);
        }

        std::shared_ptr<Obj> AstFloat::eval(EvalCtx& ctx) const
        {
            return create_obj<ObjType::FLOAT>(val);
        }

        std::shared_ptr<Obj> AstStr::eval(EvalCtx& ctx) const
        {
            return create_obj<ObjType::STR>(val);
        }

        std::shared_ptr<Obj> AstIdn::eval(EvalCtx& ctx) const
        {
            if (ctx.contains(idn))
                return ctx[idn];
            throw GenerationError("Unable to find a variable with name " + idn);
        }

        std::shared_ptr<Obj> AstTuple::eval(EvalCtx& ctx) const
        {
            std::vector<std::shared_ptr<Obj>> obj_elems;
            for (auto e : this->elems)
                e->append_vec(ctx, obj_elems);
            return create_obj<ObjType::TUPLE>(obj_elems);
        }

        std::shared_ptr<Obj> AstCall::eval(EvalCtx& ctx) const
        {
            std::shared_ptr<Obj> pobj = pleft->eval(ctx);
            std::vector<std::shared_ptr<Obj>> obj_args;
            for (auto e : this->args)
                e->append_vec(ctx, obj_args);
            return pobj->call(ctx, obj_args);;
        }

        std::shared_ptr<Obj> AstCargs::eval(EvalCtx& ctx) const
        {
            std::shared_ptr<Obj> pobj = pleft->eval(ctx);
            std::vector<std::shared_ptr<Obj>> obj_args;
            for (auto e : this->args)
                e->append_vec(ctx, obj_args);
            return pobj->cargs(obj_args);
        }

        std::shared_ptr<Obj> AstIdx::eval(EvalCtx& ctx) const
        {

        }

        std::shared_ptr<Obj> AstDot::eval(EvalCtx& ctx) const
        {

        }

        std::shared_ptr<Obj> AstNeg::eval(EvalCtx& ctx) const
        {

        }

        std::shared_ptr<Obj> AstPack::eval(EvalCtx& ctx) const
        {

        }

        std::shared_ptr<Obj> AstAdd::eval(EvalCtx& ctx) const
        {

        }

        std::shared_ptr<Obj> AstSub::eval(EvalCtx& ctx) const
        {

        }

        std::shared_ptr<Obj> AstMul::eval(EvalCtx& ctx) const
        {

        }

        std::shared_ptr<Obj> AstDiv::eval(EvalCtx& ctx) const
        {

        }

        std::shared_ptr<Obj> AstEq::eval(EvalCtx& ctx) const
        {

        }

        std::shared_ptr<Obj> AstNe::eval(EvalCtx& ctx) const
        {

        }

        std::shared_ptr<Obj> AstGe::eval(EvalCtx& ctx) const
        {

        }

        std::shared_ptr<Obj> AstLe::eval(EvalCtx& ctx) const
        {

        }

        std::shared_ptr<Obj> AstGt::eval(EvalCtx& ctx) const
        {

        }

        std::shared_ptr<Obj> AstLt::eval(EvalCtx& ctx) const
        {

        }

        std::shared_ptr<Obj> AstAnd::eval(EvalCtx& ctx) const
        {

        }

        std::shared_ptr<Obj> AstOr::eval(EvalCtx& ctx) const
        {

        }

        std::shared_ptr<Obj> AstIAdd::eval(EvalCtx& ctx) const
        {

        }

        std::shared_ptr<Obj> AstISub::eval(EvalCtx& ctx) const
        {

        }

        std::shared_ptr<Obj> AstIMul::eval(EvalCtx& ctx) const
        {

        }

        std::shared_ptr<Obj> AstIDiv::eval(EvalCtx& ctx) const
        {

        }

        std::shared_ptr<Obj> AstAssign::eval(EvalCtx& ctx) const
        {

        }

        std::shared_ptr<Obj> AstDecl::eval(EvalCtx& ctx) const
        {

        }

        std::shared_ptr<Obj> AstSeq::eval(EvalCtx& ctx) const
        {

        }

        std::shared_ptr<Obj> AstIf::eval(EvalCtx& ctx) const
        {

        }

        std::shared_ptr<Obj> AstWhile::eval(EvalCtx& ctx) const
        {

        }

        std::shared_ptr<Obj> AstFor::eval(EvalCtx& ctx) const
        {

        }

        void AstDef::eval(EvalCtx& ctx) const
        {

        }

        void AstIntr::eval(EvalCtx& ctx) const
        {

        }

        void AstFn::eval(EvalCtx& ctx) const
        {

        }

        void AstModImp::eval(EvalCtx& ctx) const
        {

        }

        EvalCtx* AstModule::eval(const std::string& entry_point, std::vector<std::shared_ptr<Obj>>& cargs)
        {
            EvalCtx* pctx = new EvalCtx();

            // Building EvalCtx
            for (auto e : imps)
                e.eval(*pctx);

            for (auto e : fns)
                e.eval(*pctx);
            for (auto e : intrs)
                e.eval(*pctx);
            for (auto e : defs)
                e.eval(*pctx);

            // Finding the entry point
            decltype(pctx->defs)::iterator it = pctx->defs.find(entry_point);
            if (it == pctx->defs.end())
                throw GenerationError("Unable to find a valid entry point with name " + entry_point);
            
            // applying cargs to the entry def
            std::shared_ptr<Obj> entry_def = it->second->cargs(cargs);

            // Generating the arguments for the entry point
            const auto& vargs = static_cast<const ObjDef*>(entry_def.get())->data.def->vargs;
            std::vector<std::shared_ptr<Obj>> args;
            for (const AstDecl& e : vargs)
            {
                if (e.type_name != "tensor")
                    throw GenerationError("'def' must have only tensor types for varargs");
                // building the arguments one by one
                args.push_back(e.eval(*pctx));
            }

            // running model generation
            entry_def->call(*pctx, args);

            return pctx;
        }
    }
}
