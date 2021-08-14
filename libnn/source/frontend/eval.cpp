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
        std::shared_ptr<Obj> last_ret = nullptr;

        std::shared_ptr<Obj> AstBool::eval(EvalCtx& ctx) const
        {
            return create_obj_bool(val);
        }

        std::shared_ptr<Obj> AstInt::eval(EvalCtx& ctx) const
        {
            return create_obj_int(val);
        }

        std::shared_ptr<Obj> AstFloat::eval(EvalCtx& ctx) const
        {
            return create_obj_float(val);
        }

        std::shared_ptr<Obj> AstStr::eval(EvalCtx& ctx) const
        {
            return create_obj_str(val);
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
            return create_obj_tuple(obj_elems);
        }

        std::shared_ptr<Obj> AstCall::eval(EvalCtx& ctx) const
        {
            std::shared_ptr<Obj> pobj = pleft->eval(ctx);
            std::vector<std::shared_ptr<Obj>> obj_args;
            for (auto e : this->args)
                e->append_vec(ctx, obj_args);
            std::shared_ptr<Obj> ret;
            return pobj->call(ctx, obj_args);
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
            return pleft->eval(ctx)->idx(pidx->eval(ctx));
            /*
            std::shared_ptr<Obj> pobj = pleft->eval(ctx);
            std::vector<std::vector<std::shared_ptr<Obj>>> obj_indicies;
            for (auto index : indicies)
            {
                obj_indicies.push_back(std::vector<std::shared_ptr<Obj>>());
                for (auto e : index)
                    obj_indicies.back().push_back(e->eval(ctx));
            }
            return pobj->idx(obj_indicies);
            */
        }

        std::shared_ptr<Obj> AstDot::eval(EvalCtx& ctx) const
        {
            return pleft->eval(ctx)->get(member);
        }

        std::shared_ptr<Obj> AstNeg::eval(EvalCtx& ctx) const
        {
            return pexpr->eval(ctx)->neg();
        }

        std::shared_ptr<Obj> AstPack::eval(EvalCtx& ctx) const
        {
            throw GenerationError("Invalid use of packed object");
        }

        std::shared_ptr<Obj> AstAdd::eval(EvalCtx& ctx) const
        {
            return pleft->eval(ctx)->add(pright->eval(ctx));
        }

        std::shared_ptr<Obj> AstSub::eval(EvalCtx& ctx) const
        {
            return pleft->eval(ctx)->sub(pright->eval(ctx));
        }

        std::shared_ptr<Obj> AstMul::eval(EvalCtx& ctx) const
        {
            return pleft->eval(ctx)->mul(pright->eval(ctx));
        }

        std::shared_ptr<Obj> AstDiv::eval(EvalCtx& ctx) const
        {
            return pleft->eval(ctx)->div(pright->eval(ctx));
        }

        std::shared_ptr<Obj> AstEq::eval(EvalCtx& ctx) const
        {
            return pleft->eval(ctx)->eq(pright->eval(ctx));
        }

        std::shared_ptr<Obj> AstNe::eval(EvalCtx& ctx) const
        {
            return pleft->eval(ctx)->ne(pright->eval(ctx));
        }

        std::shared_ptr<Obj> AstGe::eval(EvalCtx& ctx) const
        {
            return pleft->eval(ctx)->ge(pright->eval(ctx));
        }

        std::shared_ptr<Obj> AstLe::eval(EvalCtx& ctx) const
        {
            return pleft->eval(ctx)->le(pright->eval(ctx));
        }

        std::shared_ptr<Obj> AstGt::eval(EvalCtx& ctx) const
        {
            return pleft->eval(ctx)->gt(pright->eval(ctx));
        }

        std::shared_ptr<Obj> AstLt::eval(EvalCtx& ctx) const
        {
            return pleft->eval(ctx)->lt(pright->eval(ctx));
        }

        std::shared_ptr<Obj> AstAnd::eval(EvalCtx& ctx) const
        {
            return pleft->eval(ctx)->andop(pright->eval(ctx));

        }

        std::shared_ptr<Obj> AstOr::eval(EvalCtx& ctx) const
        {
            return pleft->eval(ctx)->orop(pright->eval(ctx));

        }

        std::shared_ptr<Obj> AstIAdd::eval(EvalCtx& ctx) const
        {
            std::shared_ptr<Obj> obj_left = pleft->eval(ctx);
            obj_left->assign(obj_left->add(pright->eval(ctx)));
            return create_obj_invalid();
        }

        std::shared_ptr<Obj> AstISub::eval(EvalCtx& ctx) const
        {
            std::shared_ptr<Obj> obj_left = pleft->eval(ctx);
            obj_left->assign(obj_left->sub(pright->eval(ctx)));
            return create_obj_invalid();
        }

        std::shared_ptr<Obj> AstIMul::eval(EvalCtx& ctx) const
        {
            std::shared_ptr<Obj> obj_left = pleft->eval(ctx);
            obj_left->assign(obj_left->mul(pright->eval(ctx)));
            return create_obj_invalid();
        }

        std::shared_ptr<Obj> AstIDiv::eval(EvalCtx& ctx) const
        {
            std::shared_ptr<Obj> obj_left = pleft->eval(ctx);
            obj_left->assign(obj_left->div(pright->eval(ctx)));
            return create_obj_invalid();
        }

        std::shared_ptr<Obj> AstAssign::eval(EvalCtx& ctx) const
        {
            pleft->eval(ctx)->assign(pright->eval(ctx));
            return create_obj_invalid();
        }

        std::shared_ptr<Obj> AstDecl::eval(EvalCtx& ctx) const
        {
            auto get_type = [](const std::string& name)
            {
                if (name == "bool")
                    return ObjType::BOOL;
                else if (name == "int")
                    return ObjType::INT;
                else if (name == "float")
                    return ObjType::FLOAT;
                else if (name == "str")
                    return ObjType::STR;
                else if (name == "array")
                    return ObjType::ARRAY;
                else if (name == "tuple")
                    return ObjType::TUPLE;
                else if (name == "tensor")
                    return ObjType::TENSOR;
                else
                    throw GenerationError("Invalid type name: " + name);
            };

            std::shared_ptr<Obj> pobj;

            if (is_static)
            {
                // generating a globally unique identifier
                std::string name = ctx.block_name + "-" + var_name;
                if (ctx.state != EvalState::DEFSEQ && ctx.state != EvalState::DEFEXPR)
                    throw GenerationError("Invalid context for static declaration");
                
                // doing the decl
                if (ctx.statics.contains(name))
                {
                    pobj = ctx.statics[name];
                    if (pobj->ty != get_type(type_name))
                        throw GenerationError("Invalid redeclaration of static variable");
                    std::vector<std::shared_ptr<Obj>> obj_cargs;
                    for (auto e : cargs)
                        obj_cargs.push_back(e->eval(ctx));
                    if (!pobj->check_cargs(obj_cargs))
                        throw GenerationError("");
                    ctx.scope()[var_name] = pobj;
                    return pobj;
                }
                else
                {
                    pobj = create_obj_type(get_type(type_name));
                    std::vector<std::shared_ptr<Obj>> obj_cargs;
                    for (auto e : cargs)
                        obj_cargs.push_back(e->eval(ctx));
                    if (obj_cargs.size() != 0)
                        pobj = pobj->cargs(obj_cargs);
                    ctx.statics[name] = pobj;
                }
            }
            else
            {
                pobj = create_obj_type(get_type(type_name));
                std::vector<std::shared_ptr<Obj>> obj_cargs;
                for (auto e : cargs)
                    obj_cargs.push_back(e->eval(ctx));
                if (obj_cargs.size() != 0)
                    pobj = pobj->cargs(obj_cargs);
            }

            // custom tensor declaration stuff for network inputs
            if (type_name == "tensor" && ctx.state == EvalState::DEFSEQ)
            {
                ObjTensor* pten = static_cast<ObjTensor*>(pobj.get());
                assert(pten->ty == ObjType::TENSOR);
                if (!pten->data.carg_init)
                    throw GenerationError("Standalone tensor declarations must have constant arguments");

                // creating a new edge
                pten->data.pEdge = new Edge();
                pten->data.pEdge->dsc.rk = pten->data.dims.size();
                for (auto e : pten->data.dims)
                    pten->data.pEdge->dsc.dims.push_back(e);

                // adding the tensor as an input edge
                if (!varcounts.contains(var_name))
                    varcounts[var_name] = 0;
                int id = varcounts[var_name]++;
                ctx.graph().inputs[var_name + '-' + std::to_string(id)] = pten->data.pEdge;
            }

            return pobj;
        }

        std::shared_ptr<Obj> AstReturn::eval(EvalCtx& ctx) const
        {
            if (last_ret)
                throw GenerationError("Invalid return statement");

            last_ret = ret->eval(ctx);
            return create_obj_invalid();
        }


        std::shared_ptr<Obj> AstSeq::eval(EvalCtx& ctx) const
        {
            for (auto e : blocks)
            {
                if (last_ret)
                    return create_obj_invalid();
                e->eval(ctx);
            }
            return create_obj_invalid();
        }

        std::shared_ptr<Obj> AstIf::eval(EvalCtx& ctx) const
        {
            // TODO: add in elif/else block
        }

        std::shared_ptr<Obj> AstWhile::eval(EvalCtx& ctx) const
        {
            if (last_ret)
                return create_obj_invalid();
            while (pcond->eval(ctx)->bval())
            {
                seq.eval(ctx);
                if (last_ret)
                    return create_obj_invalid();
            }
            return create_obj_invalid();
        }

        std::shared_ptr<Obj> AstFor::eval(EvalCtx& ctx) const
        {
            if (last_ret)
                return create_obj_invalid();
            std::shared_ptr<Obj> idx = pexpr->eval(ctx);
            for (auto e : it.eval(ctx)->iter(ctx))
            {
                idx->assign(e);
                seq.eval(ctx);
                if (last_ret)
                    return create_obj_invalid();
            }
            return create_obj_invalid();
        }

        void AstDef::eval(EvalCtx& ctx) const
        {
            ctx.defs.insert({ name, create_obj_def(this) });
        }

        void AstIntr::eval(EvalCtx& ctx) const
        {
            ctx.defs.insert({ name, create_obj_intr(this) });
        }

        void AstFn::eval(EvalCtx& ctx) const
        {
            ctx.defs.insert({ name, create_obj_fn(this) });
        }

        void AstModImp::eval(EvalCtx& ctx) const
        {
            // TODO: figure out module/package importing
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
            const auto& vargs = static_cast<const ObjDef*>(entry_def.get())->data.pdef->vargs;
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
