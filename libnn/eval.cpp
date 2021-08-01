#include <libnn/frontend/eval.h>
#include <libnn/frontend/obj.h>
#include <libnn/frontend/ast.h>

#include <cassert>

namespace nn
{
    namespace impl
    {
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

        Scope::Scope() :
            data({})
        {}

        Scope::~Scope()
        {
            for (auto& [k, v] : this->data)
                delete v;
        }

        Obj* Scope::operator[](const std::string& idn)
        {
            return data[idn];
        }

        EvalCtx::EvalCtx()
        {
        }



        EvalCtx* AstModule::eval(const std::string& entry_point, std::vector<Obj*>& args)
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

            // Generating the model
            decltype(pctx->defs)::iterator it = pctx->defs.find(entry_point);
            if (it == pctx->defs.end())
                throw GenerationError("Unable to find a valid entry point with name " + entry_point);
            Obj* entry_def = it->second;
            entry_def = entry_def->cargs(args);

            const auto& varargs = static_cast<ObjDef*>(entry_def)->getData().def.varargs;
            for (auto& e : varargs)
            {
                if (e.type_name != "tensor")
                    throw GenerationError("'def' must have only tensor types for varargs");
                // TODO: make the input tensor nodes
            }

            // TODO: entry_def->call()
        }
    }
}
