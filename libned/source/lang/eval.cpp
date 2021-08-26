#include <ned/lang/eval.h>
#include <ned/lang/obj.h>
#include <ned/lang/ast.h>

#include <cassert>

namespace nn
{
    namespace impl
    {
        ObjType dec_typename_exc(const std::string& name)
        {
            if (name == "var")
                return ObjType::VAR;
            else if (name == "bool")
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

        ObjType dec_typename_inv(const std::string& name) noexcept
        {
            if (name == "var")
                return ObjType::VAR;
            else if (name == "bool")
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
                return ObjType::INVALID;
        };

        // Definitions for eval.h

        void AstExpr::append_vec(EvalCtx& ctx, std::vector<std::shared_ptr<Obj>>& cargs) const
        {
            cargs.push_back(eval(ctx));
        }

        void AstPack::append_vec(EvalCtx& ctx, std::vector<std::shared_ptr<Obj>>& cargs) const
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

        EvalCtx::~EvalCtx()
        {
            delete pgraph;
            delete pscope;

            for (auto e : defs)
                delete std::static_pointer_cast<ObjDef>(std::get<1>(e))->data.pdef;
            for (auto e : fns)
                delete std::static_pointer_cast<ObjFn>(std::get<1>(e))->data.pfn;
            for (auto e : intrs)
                delete std::static_pointer_cast<ObjIntr>(std::get<1>(e))->data.pintr;
        }

        std::shared_ptr<Obj> EvalCtx::get(const std::string& name)
        {
            if (pscope->contains(name))
                return pscope->at(name);
            if (defs.contains(name))
                return defs.at(name);
            if (fns.contains(name))
                return fns.at(name);
            if (intrs.contains(name))
                return intrs.at(name);
            throw GenerationError("Unable to resolve identifier '" + name + "'");
        }

        bool EvalCtx::contains(const std::string& name) const noexcept
        {
            if (pscope->contains(name))
                return true;
            if (defs.contains(name))
                return true;
            if (fns.contains(name))
                return true;
            if (intrs.contains(name))
                return true;
            return false;
        }

        Scope& EvalCtx::scope() noexcept
        {
            return *pscope;
        }

        Graph& EvalCtx::graph() noexcept
        {
            return *pgraph;
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
            // checking for types
            ObjType ty = dec_typename_inv(idn);
            if (ty != ObjType::INVALID)
                return create_obj_dtype(ty);

            // regular identifier stuff
            return ctx.get(idn);
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
            EvalState current_state = ctx.state;
            ctx.state = EvalState::CALL;
            std::vector<std::shared_ptr<Obj>> obj_args;
            for (auto e : this->args)
                e->append_vec(ctx, obj_args);
            ctx.state = current_state;
            pobj->call(ctx, obj_args);
            std::shared_ptr<Obj> pret(nullptr);
            pret.swap(last_ret);
            return pret;
        }

        std::shared_ptr<Obj> AstCargs::eval(EvalCtx& ctx) const
        {
            std::shared_ptr<Obj> pobj = pleft->eval(ctx);
            EvalState current_state = ctx.state;
            ctx.state = EvalState::CALL;
            std::vector<std::shared_ptr<Obj>> obj_args;
            for (auto e : this->args)
                e->append_vec(ctx, obj_args);
            ctx.state = current_state;
            return pobj->cargs(obj_args);
        }

        std::shared_ptr<Obj> AstIdx::eval(EvalCtx& ctx) const
        {
            bool revert = false;
            if (ctx.state == EvalState::DEFSEQ)
            {
                ctx.state = EvalState::DEFEXPR;
                revert = true;
            }
            std::shared_ptr<Obj> pret = pleft->eval(ctx)->idx(pidx->eval(ctx));
            if (revert)
                ctx.state = EvalState::DEFSEQ;
            return pret;
        }

        std::shared_ptr<Obj> AstDot::eval(EvalCtx& ctx) const
        {
            bool revert = false;
            if (ctx.state == EvalState::DEFSEQ)
            {
                ctx.state = EvalState::DEFEXPR;
                revert = true;
            }
            std::shared_ptr<Obj> pret = pleft->eval(ctx)->get(member);
            if (revert)
                ctx.state = EvalState::DEFSEQ;
            return pret;
        }

        std::shared_ptr<Obj> AstNeg::eval(EvalCtx& ctx) const
        {
            bool revert = false;
            if (ctx.state == EvalState::DEFSEQ)
            {
                ctx.state = EvalState::DEFEXPR;
                revert = true;
            }
            std::shared_ptr<Obj> pret = pexpr->eval(ctx)->neg();
            if (revert)
                ctx.state = EvalState::DEFSEQ;
            return pret;
        }

        std::shared_ptr<Obj> AstPack::eval(EvalCtx& ctx) const
        {
            return pexpr->eval(ctx);
        }

        std::shared_ptr<Obj> AstAdd::eval(EvalCtx& ctx) const
        {
            bool revert = false;
            if (ctx.state == EvalState::DEFSEQ)
            {
                ctx.state = EvalState::DEFEXPR;
                revert = true;
            }
            std::shared_ptr<Obj> pret = pleft->eval(ctx)->add(pright->eval(ctx));
            if (revert)
                ctx.state = EvalState::DEFSEQ;
            return pret;
        }

        std::shared_ptr<Obj> AstSub::eval(EvalCtx& ctx) const
        {
            bool revert = false;
            if (ctx.state == EvalState::DEFSEQ)
            {
                ctx.state = EvalState::DEFEXPR;
                revert = true;
            }
            std::shared_ptr<Obj> pret = pleft->eval(ctx)->sub(pright->eval(ctx));
            if (revert)
                ctx.state = EvalState::DEFSEQ;
            return pret;
        }

        std::shared_ptr<Obj> AstMul::eval(EvalCtx& ctx) const
        {
            bool revert = false;
            if (ctx.state == EvalState::DEFSEQ)
            {
                ctx.state = EvalState::DEFEXPR;
                revert = true;
            }
            std::shared_ptr<Obj> pret = pleft->eval(ctx)->mul(pright->eval(ctx));
            if (revert)
                ctx.state = EvalState::DEFSEQ;
            return pret;
        }

        std::shared_ptr<Obj> AstDiv::eval(EvalCtx& ctx) const
        {
            bool revert = false;
            if (ctx.state == EvalState::DEFSEQ)
            {
                ctx.state = EvalState::DEFEXPR;
                revert = true;
            }
            std::shared_ptr<Obj> pret = pleft->eval(ctx)->div(pright->eval(ctx));
            if (revert)
                ctx.state = EvalState::DEFSEQ;
            return pret;
        }

        std::shared_ptr<Obj> AstEq::eval(EvalCtx& ctx) const
        {
            bool revert = false;
            if (ctx.state == EvalState::DEFSEQ)
            {
                ctx.state = EvalState::DEFEXPR;
                revert = true;
            }
            std::shared_ptr<Obj> pret = pleft->eval(ctx)->eq(pright->eval(ctx));
            if (revert)
                ctx.state = EvalState::DEFSEQ;
            return pret;
        }

        std::shared_ptr<Obj> AstNe::eval(EvalCtx& ctx) const
        {
            bool revert = false;
            if (ctx.state == EvalState::DEFSEQ)
            {
                ctx.state = EvalState::DEFEXPR;
                revert = true;
            }
            std::shared_ptr<Obj> pret = pleft->eval(ctx)->ne(pright->eval(ctx));
            if (revert)
                ctx.state = EvalState::DEFSEQ;
            return pret;
        }

        std::shared_ptr<Obj> AstGe::eval(EvalCtx& ctx) const
        {
            bool revert = false;
            if (ctx.state == EvalState::DEFSEQ)
            {
                ctx.state = EvalState::DEFEXPR;
                revert = true;
            }
            std::shared_ptr<Obj> pret = pleft->eval(ctx)->ge(pright->eval(ctx));
            if (revert)
                ctx.state = EvalState::DEFSEQ;
            return pret;
        }

        std::shared_ptr<Obj> AstLe::eval(EvalCtx& ctx) const
        {
            bool revert = false;
            if (ctx.state == EvalState::DEFSEQ)
            {
                ctx.state = EvalState::DEFEXPR;
                revert = true;
            }
            std::shared_ptr<Obj> pret = pleft->eval(ctx)->le(pright->eval(ctx));
            if (revert)
                ctx.state = EvalState::DEFSEQ;
            return pret;
        }

        std::shared_ptr<Obj> AstGt::eval(EvalCtx& ctx) const
        {
            bool revert = false;
            if (ctx.state == EvalState::DEFSEQ)
            {
                ctx.state = EvalState::DEFEXPR;
                revert = true;
            }
            std::shared_ptr<Obj> pret = pleft->eval(ctx)->gt(pright->eval(ctx));
            if (revert)
                ctx.state = EvalState::DEFSEQ;
            return pret;
        }

        std::shared_ptr<Obj> AstLt::eval(EvalCtx& ctx) const
        {
            bool revert = false;
            if (ctx.state == EvalState::DEFSEQ)
            {
                ctx.state = EvalState::DEFEXPR;
                revert = true;
            }
            std::shared_ptr<Obj> pret = pleft->eval(ctx)->lt(pright->eval(ctx));
            if (revert)
                ctx.state = EvalState::DEFSEQ;
            return pret;
        }

        std::shared_ptr<Obj> AstAnd::eval(EvalCtx& ctx) const
        {
            bool revert = false;
            if (ctx.state == EvalState::DEFSEQ)
            {
                ctx.state = EvalState::DEFEXPR;
                revert = true;
            }
            std::shared_ptr<Obj> pret = pleft->eval(ctx)->andop(pright->eval(ctx));
            if (revert)
                ctx.state = EvalState::DEFSEQ;
            return pret;
        }

        std::shared_ptr<Obj> AstOr::eval(EvalCtx& ctx) const
        {
            bool revert = false;
            if (ctx.state == EvalState::DEFSEQ)
            {
                ctx.state = EvalState::DEFEXPR;
                revert = true;
            }
            std::shared_ptr<Obj> pret = pleft->eval(ctx)->orop(pright->eval(ctx));
            if (revert)
                ctx.state = EvalState::DEFSEQ;
            return pret;
        }

        std::shared_ptr<Obj> AstIAdd::eval(EvalCtx& ctx) const
        {
            bool revert = false;
            if (ctx.state == EvalState::DEFSEQ)
            {
                ctx.state = EvalState::DEFEXPR;
                revert = true;
            }
            std::shared_ptr<Obj> obj_left = pleft->eval(ctx);
            obj_left->assign(obj_left->add(pright->eval(ctx)));
            if (revert)
                ctx.state = EvalState::DEFSEQ;
            return create_obj_invalid();
        }

        std::shared_ptr<Obj> AstISub::eval(EvalCtx& ctx) const
        {
            bool revert = false;
            if (ctx.state == EvalState::DEFSEQ)
            {
                ctx.state = EvalState::DEFEXPR;
                revert = true;
            }
            std::shared_ptr<Obj> obj_left = pleft->eval(ctx);
            obj_left->assign(obj_left->sub(pright->eval(ctx)));
            if (revert)
                ctx.state = EvalState::DEFSEQ;
            return create_obj_invalid();
        }

        std::shared_ptr<Obj> AstIMul::eval(EvalCtx& ctx) const
        {
            bool revert = false;
            if (ctx.state == EvalState::DEFSEQ)
            {
                ctx.state = EvalState::DEFEXPR;
                revert = true;
            }
            std::shared_ptr<Obj> obj_left = pleft->eval(ctx);
            obj_left->assign(obj_left->mul(pright->eval(ctx)));
            if (revert)
                ctx.state = EvalState::DEFSEQ;
            return create_obj_invalid();
        }

        std::shared_ptr<Obj> AstIDiv::eval(EvalCtx& ctx) const
        {
            bool revert = false;
            if (ctx.state == EvalState::DEFSEQ)
            {
                ctx.state = EvalState::DEFEXPR;
                revert = true;
            }
            std::shared_ptr<Obj> obj_left = pleft->eval(ctx);
            obj_left->assign(obj_left->div(pright->eval(ctx)));
            if (revert)
                ctx.state = EvalState::DEFSEQ;
            return create_obj_invalid();
        }

        std::shared_ptr<Obj> AstAssign::eval(EvalCtx& ctx) const
        {
            bool revert = false;
            if (ctx.state == EvalState::DEFSEQ)
            {
                ctx.state = EvalState::DEFEXPR;
                revert = true;
            }
            pleft->eval(ctx)->assign(pright->eval(ctx));
            if (revert)
                ctx.state = EvalState::DEFSEQ;
            return create_obj_invalid();
        }

        std::shared_ptr<Obj> AstDecl::eval(EvalCtx& ctx) const
        {
            std::shared_ptr<Obj> dtype = type_idn.eval(ctx);
            if (has_cargs)
            {
                std::vector<std::shared_ptr<Obj>> obj_cargs;
                for (auto e : cargs)
                    obj_cargs.push_back(e->eval(ctx));
                dtype = dtype->cargs(obj_cargs);
            }
            std::shared_ptr<Obj> pobj = dtype->inst();

            if (is_static)
            {
                // generating a globally unique identifier
                std::string name = ctx.block_name + "-" + var_name;
                if (ctx.state != EvalState::DEFSEQ && ctx.state != EvalState::DEFEXPR)
                    throw GenerationError("Invalid context for static declaration");
                
                // doing the decl
                if (ctx.statics.contains(name))
                {
                    pobj->assign(ctx.statics[name]);
                    ctx.scope()[var_name] = pobj;
                    return pobj;
                }
                else
                    ctx.statics[name] = pobj;
            }

            // custom tensor declaration stuff for network inputs
            if (pobj->ty == ObjType::TENSOR && ctx.state == EvalState::DEFSEQ)
            {
                ObjTensor* pten = static_cast<ObjTensor*>(pobj.get());
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

        std::shared_ptr<Obj> AstPrint::eval(EvalCtx& ctx) const
        {
            if (last_ret)
                return create_obj_invalid();
            printf(val->eval(ctx)->str().c_str());
            return create_obj_invalid();
        }

        std::shared_ptr<Obj> AstReturn::eval(EvalCtx& ctx) const
        {
            if (last_ret)
                throw GenerationError("Invalid return statement");

            last_ret = ret->eval(ctx);
            return create_obj_invalid();
        }

        std::shared_ptr<Obj> AstRaise::eval(EvalCtx& ctx) const
        {
            if (last_ret)
                return create_obj_invalid();
            throw GenerationError(val->eval(ctx)->str());
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
            throw GenerationError("Not implemented");
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

        std::vector<std::shared_ptr<Obj>>::iterator AstCargDecl::match_args(
            EvalCtx& ctx,
            std::vector<std::shared_ptr<Obj>>::iterator start,
            std::vector<std::shared_ptr<Obj>>::iterator end) const
        {
            if (ctx.scope().contains(var_name))
                throw GenerationError("Constant args name collision");

            // getting the dtype (same as normal decl)
            std::shared_ptr<Obj> dtype = type_idn.eval(ctx);
            if (has_cargs)
            {
                std::vector<std::shared_ptr<Obj>> obj_cargs;
                for (auto e : cargs)
                    obj_cargs.push_back(e->eval(ctx));
                dtype = dtype->cargs(obj_cargs);
            }

            // No values given, wait for argument type deduction
            if (start == end)
            {
                if (is_packed)
                    ctx.scope()[var_name] = create_obj_array(dtype, 0);
                else
                    ctx.scope()[var_name] = dtype->inst();
                return start;
            }

            if (is_packed)
            {
                // greedly match the arguments
                std::vector<std::shared_ptr<Obj>> matches;
                while (start != end && dtype->eq(*start)->bval())
                {
                    auto pobj = dtype->inst();
                    pobj->assign(*start);
                    matches.push_back(pobj);
                    start++;
                }
                ctx.scope()[var_name] = create_obj_array(dtype, matches);
                return start;
            }

            // match exactly one element
            ctx.scope()[var_name] = dtype->inst();
            ctx.scope()[var_name]->assign(*start);
            return ++start;
        }

        std::vector<std::shared_ptr<Obj>>::iterator AstCargTuple::match_args(
            EvalCtx& ctx,
            std::vector<std::shared_ptr<Obj>>::iterator start,
            std::vector<std::shared_ptr<Obj>>::iterator end) const
        {
            std::vector<AstCargSig*>::const_iterator it;
            if (start == end)
            {
                for (it = elems.begin(); it != elems.end(); it++)
                    assert((*it)->match_args(ctx, start, end) == start);
                return start;
            }
            if ((*start)->ty != ObjType::TUPLE)
                throw GenerationError("Invalid carg type to match tuple");

            auto tbeg = static_cast<ObjTuple*>((*start).get())->data.elems.begin();
            auto tend = static_cast<ObjTuple*>((*start).get())->data.elems.end();
            for (it = elems.begin(); it != elems.end(); it++)
                tbeg = (*it)->match_args(ctx, tbeg, tend);
            if (it != elems.end())
                throw GenerationError("Too many carg initializer values");
            return ++start;
        }

        AstArgSig::Iter AstArgImm::carg_deduction(EvalCtx& ctx, const AstArgSig::Iter& start, const AstArgSig::Iter& end) const
        {
            if (start == end)
                throw GenerationError("Missing data for carg deduction");
            if (pimm->eval(ctx)->ne(*start)->bval())
                throw GenerationError("Argument data mismatch");
            return std::next(start);
        }

        AstArgSig::Iter AstArgVar::carg_deduction(EvalCtx& ctx, const AstArgSig::Iter& start, const AstArgSig::Iter& end) const
        {
            if (!ctx.scope().contains(var_name))
                throw GenerationError("Unable to find carg parameter " + var_name);
            std::shared_ptr<Obj> carg_obj = ctx.scope()[var_name];

            if (is_packed)
            {
                assert(carg_obj->ty == ObjType::ARRAY);
                
                auto it = start;
                auto arr_obj = std::static_pointer_cast<ObjArray>(carg_obj);
                if (arr_obj->data.elems.size() != 0)
                {
                    // It is initialized; don't deduce, just check
                    for (auto e : arr_obj->data.elems)
                    {
                        if (it == end)
                            throw GenerationError("Not enough cargs in call to match the carg parameters");
                        if (e->ne(*it)->bval())
                            throw GenerationError("Value mismatch between call cargs and parameter cargs");
                    }
                    return it;
                }
                while (it != end)
                {
                    try
                    {
                        std::shared_ptr<Obj> inst = arr_obj->data.dtype->inst();
                        inst->assign(*start);
                        arr_obj->data.elems.push_back(inst);
                    }
                    catch (const GenerationError& e)
                    {
                        // Not an error, it just means the end of the packed parameters
                        return it;
                    }
                }
            }
            else
            {
                if (start == end)
                    throw GenerationError("Missing data for carg deduction");
                if (carg_obj->init)
                {
                    if (carg_obj->ne(*start)->bval())
                        throw GenerationError("Argument value mismatch in carg var deduction");
                }
                else
                    carg_obj->assign(*start);
                return std::next(start);
            }
        }

        AstArgSig::Iter AstArgDecl::carg_deduction(EvalCtx& ctx, const AstArgSig::Iter& start, const AstArgSig::Iter& end) const
        {
            if (start == end)
                throw GenerationError("Missing data for carg deduction");
            if (dec_typename_exc(type_name) != (*start)->ty)
                throw GenerationError("type mismatch, expected " + type_name + ", recieved " + obj_type_name((*start)->ty));
            if (!has_cargs)
                return std::next(start);
            std::vector<std::shared_ptr<Obj>> elems = (*start)->iter(ctx);
            auto e_start = elems.begin();
            auto e_end = elems.end();
            for (auto e : cargs)
                e_start = e->carg_deduction(ctx, e_start, e_end);
            if (e_start != e_end)
                throw GenerationError("Too many elements in the passed argument relative to its arg decl");
            return std::next(start);
        }

        void AstDef::eval(EvalCtx& ctx) const
        {
            ctx.defs.insert({ name, create_obj_def(this) });
        }

        void AstDef::apply_cargs(EvalCtx& ctx, std::vector<std::shared_ptr<Obj>>& cargs) const
        {
            std::vector<std::shared_ptr<Obj>> cargs_tuple = { create_obj_tuple(cargs) };
            assert(this->cargs->match_args(ctx, cargs_tuple.begin(), cargs_tuple.end()) == cargs_tuple.end());
        }

        void AstDef::carg_deduction(EvalCtx& ctx, std::vector<std::shared_ptr<Obj>>& args) const
        {
            if (args.size() != vargs.size())
                throw GenerationError("Argument count mismatch, expected " + std::to_string(vargs.size()) + ", recieved " + std::to_string(args.size()));
            for (int i = 0; i < args.size(); i++)
            {
                std::shared_ptr<Obj> arg_cpy = args[i]->copy();
                std::vector<std::shared_ptr<Obj>> arg_vec = { arg_cpy };
                assert(std::get<0>(vargs[i]).carg_deduction(ctx, arg_vec.begin(), arg_vec.end()) == arg_vec.end());
                if (ctx.scope().contains(std::get<1>(vargs[i])))
                    throw GenerationError("Name collision in def call");
                ctx.scope()[std::get<1>(vargs[i])] = arg_cpy;
            }
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
            throw GenerationError("Not implemented");
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
            for (auto& [decl, name] : vargs)
                args.push_back(decl.auto_gen(*pctx, name));  // automatically generating the arguments one by one

            // running model generation
            entry_def->call(*pctx, args);

            // TODO: add last_ret to the model outputs
            throw GenerationError("Not implemented");

            return pctx;
        }
    }
}
