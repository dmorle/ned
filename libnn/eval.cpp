#include <libnn/frontend/eval.h>
#include <libnn/frontend/obj.h>
#include <libnn/frontend/ast.h>

#include <cassert>

namespace nn
{
    namespace impl
    {
        void AstExpr::append_cargs(EvalCtx& ctx, std::vector<std::unique_ptr<Obj>>& cargs) const
        {
            cargs.push_back(eval(ctx));
        }

        void AstPack::append_cargs(EvalCtx& ctx, std::vector<std::unique_ptr<Obj>>& cargs)
        {
            std::vector<std::unique_ptr<Obj>> eval_result = eval(ctx)->iter(ctx);
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
            defs = {};
            fns = {};
            intrs = {};
            mods = {};
            packs = {};

            model_params = {};
            pgraph = new Graph();
            pscope = nullptr;

            state = EvalState::CALL;
        }

        std::unique_ptr<EvalCtx> AstModule::eval(const std::string& entry_point, std::vector<std::unique_ptr<Obj>>& cargs)
        {
            std::unique_ptr<EvalCtx> pctx = std::make_unique<EvalCtx>();

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
            std::unique_ptr<Obj> entry_def = std::move(it->second);
            
            // applying cargs to the entry def
            entry_def = entry_def->cargs(cargs);

            // Checking to make sure all the cargs were initialized
            for (const auto& [k, v] : static_cast<ObjDef*>(entry_def.get())->getData().cargs)
                if (!v)
                    throw GenerationError("Unable to initialize carg " + k + " in entry point");

            // Generating the arguments for the entry point
            const auto& vargs = static_cast<ObjDef*>(entry_def.get())->getData().def.vargs;
            std::vector<std::unique_ptr<Obj>> args;
            for (const AstDecl& e : vargs)
            {
                if (e.type_name != "tensor")
                    throw GenerationError("'def' must have only tensor types for varargs");
                // building the arguments one by one
                args.push_back(e.eval(*pctx));
                const auto& data = static_cast<const ObjTensor*>(args.back().get())->getData();
            }

            // running model generation
            entry_def->call(*pctx, args);

            return pctx;
        }
    }
}
