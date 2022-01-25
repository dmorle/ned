#include <ned/errors.h>
#include <ned/lang/interp.h>
#include <ned/lang/bytecode.h>

#include <iostream>

#define oprand (*((size_t*)(code + pc)))

namespace nn
{
    namespace lang
    {
        // state of the interpreter (no, this shouldn't be implemented as a singleton.)
        // programming state
        size_t pc = 0;
        CodeSegPtr code;
        DataSegPtr data;
        bool complete;
        std::vector<size_t> pc_stack;

        // deep learning state
        GraphBuilder* pbuilder;
        std::vector<std::string> md_stack;

        // graph builder operations

        GraphBuilder::GraphBuilder() {}

        bool GraphBuilder::get_forward(Obj& obj, core::Tensor* pten)
        {
            obj.edge_obj = pten->forward;
            if (obj.edge_obj)
                return error::runtime("Attempted to retrieve the uninitialized forward edge of a tensor");
            return false;
        }

        bool GraphBuilder::get_backward(Obj& obj, core::Tensor* pten)
        {
            obj.edge_obj = pten->backward;
            if (obj.edge_obj)
                return error::runtime("Attempted to retrieve the uninitialized backward edge of a tensor");
            return false;
        }

        bool GraphBuilder::add_ndcfg(const std::string& name, core::Node* pnode, core::Config* pconfig)
        {
            if (pnode->configs.contains(name))
                return error::runtime("Attempted to overwrite node configuration '%'", name);
            pnode->configs[name] = std::unique_ptr<core::Config>(pconfig);
            return false;
        }

        bool GraphBuilder::add_bkcfg(const std::string& name, core::Block* pblock, core::Config* pconfig)
        {
            if (pblock->configs.contains(name))
                return error::runtime("Attempted to overwrite block configuration '%'", name);
            pblock->configs[name] = std::unique_ptr<core::Config>(pconfig);
            return false;
        }

        bool GraphBuilder::add_incfg(const std::string& name, core::Init* pinit, core::Config* pconfig)
        {
            if (pinit->configs.contains(name))
                return error::runtime("Attempted to overwrite init configuration '%'", name);
            pinit->configs[name] = std::unique_ptr<core::Config>(pconfig);
            return false;
        }

        bool GraphBuilder::set_ndinp(const std::string& name, core::Node* pnode, core::Edge* pedge)
        {
            if (pnode->inps.contains(name))
                return error::runtime("Attempted to bind node input '%' multiple times", name);
            pnode->inps[name] = pedge;
            return false;
        }

        bool GraphBuilder::set_ndout(const std::string& name, core::Node* pnode, core::Edge* pedge)
        {
            if (pnode->outs.contains(name))
                return error::runtime("Attempted to bind node output '%' multiple times", name);
            pnode->outs[name] = pedge;
            return false;
        }

        bool GraphBuilder::set_bkinp(const std::string& name, core::Block* pblock, core::Tensor* pten)
        {
            if (pblock->inps.contains(name))
                return error::runtime("Attempted to bind block input '%' multiple times", name);
            if (!pten->forward)
                return error::runtime("Attempted to bind block input '%' to a tensor with an uninitialized forward edge", name);
            if (!pten->backward)
                return error::runtime("Attempted to bind block input '%' to a tensor with an uninitialized backward edge", name);
            pblock->inps[name] = { pten->forward, pten->backward };
            return false;
        }

        bool GraphBuilder::set_bkout(const std::string& name, core::Block* pblock, core::Tensor* pten)
        {
            if (pblock->outs.contains(name))
                return error::runtime("Attempted to bind block output '%' multiple times", name);
            if (!pten->forward)
                return error::runtime("Attempted to bind block output '%' to a tensor with an uninitialized forward edge", name);
            if (!pten->backward)
                return error::runtime("Attempted to bind block output '%' to a tensor with an uninitialized backward edge", name);
            pblock->outs[name] = { pten->forward, pten->backward };
            return false;
        }

        bool GraphBuilder::set_bkprt(core::Block* pblock, core::Block* pparent)
        {
            if (pblock->parent)
                return error::runtime("Attempted to set a block's parent when it has already been set");
            else if (pblock == root)
                return error::runtime("Attempted to set the root block's parent");

            pblock->parent = pparent;
            if (pparent)
                return false;

            if (root)
                return error::runtime("Attempted to set a block as the root, when a root was already set");
            root = pblock;
            return false;
        }

        bool GraphBuilder::add_extern(const std::string& name, core::Block* pblock, core::Tensor* pten, core::Init* pinit)
        {
            if (pblock->weights.contains(name))
                return error::runtime("Attempted to overwrite block weight '%'", name);
            if (!pten->forward)
                return error::runtime("Attempted to extern a tensor with an uninitialized forward edge");
            if (!pten->backward)
                return error::runtime("Attempted to extern a tensor with an uninitialized backward edge");
            pblock->weights[name] = { pten->forward, pten->backward, pinit, nullptr };
            return false;
        }

        bool GraphBuilder::add_export(const std::string& name, core::Tensor* pten)
        {
            if (exports.contains(name))
                return error::runtime("Attempted to overwrite model export '%'", name);
            if (!pten->forward)
                return error::runtime("Attempted to export a tensor with an uninitialized forward edge");
            if (!pten->backward)
                return error::runtime("Attempted to export a tensor with an uninitialized backward edge");
            exports[name] = { pten->forward, pten->backward };
            return false;
        }

        bool GraphBuilder::export_graph(core::Graph& graph)
        {
            // TODO: implement graph exporting
            return error::runtime("Graph exporting has not been implemented yet...");
        }

        // stack operations

        bool CallStack::pop(Obj& obj)
        {
            if (sp == 0)
                return error::runtime("Stack pointer out of bounds during pop operation");
            obj = stack[--sp];
            return false;
        }

        bool CallStack::del(size_t i)
        {
            if (i >= sp)
                return error::runtime("Attempted to delete a non-existent stack element");
            sp--;
            for (size_t j = sp - i; j < sp; j++)
                stack[j] = stack[j + 1];
            return false;
        }

        bool CallStack::get(size_t i, Obj& obj)
        {
            if (i >= sp)
                return error::runtime("Attempted to retrieve a non-existent stack element");
            obj = stack[sp - i - 1];
            return false;
        }

        bool CallStack::push(Obj obj)
        {
            if (sp >= stack.size())
                return error::runtime("Stack overflow error");
            stack[sp++] = obj;
            return false;
        }

        // helper functions

        inline bool set_pc(size_t val)
        {
            pc = val;
            return false;
        }

        inline bool push_pc()
        {
            pc_stack.push_back(pc);
            return false;
        }

        inline bool pop_pc()
        {
            assert(pc_stack.size() > 0);
            pc = pc_stack.back();
            pc_stack.pop_back();
            return false;
        }

        inline bool push_md(const std::string& mode)
        {
            md_stack.push_back(mode);
            return false;
        }

        inline bool pop_md()
        {
            if (md_stack.size() == 0)
                return error::runtime("Attempted to release a non-existent evaluation mode");
            md_stack.pop_back();
            return false;
        }

        // instruction implementations

        inline bool exec_jmp()
        {
            return
                set_pc(oprand);
        }

        inline bool exec_brt(CallStack& stack)
        {
            Obj obj;
            return
                stack.pop(obj) ||
                set_pc(*obj.bool_obj ? oprand : pc + sizeof(size_t));
        }

        inline bool exec_brf(CallStack& stack)
        {
            Obj obj;
            return
                stack.pop(obj) ||
                set_pc(*obj.bool_obj ? pc + sizeof(size_t) : oprand);
        }

        inline bool exec_new(CallStack& stack)
        {
            return
                stack.push(data[oprand]) ||
                set_pc(pc + sizeof(size_t));
        }

        inline bool exec_agg(CallStack& stack, ProgramHeap& heap)
        {
            std::vector<Obj> objs(oprand);
            for (size_t i = 0; i < oprand; i++)
            {
                Obj obj;
                if (stack.pop(obj))
                    return true;
                objs.push_back(obj);
            }
            Obj dst;
            return
                heap.create_obj_agg(dst, objs) ||
                stack.push(dst) ||
                set_pc(pc + sizeof(size_t));
        }

        inline bool exec_arr(CallStack& stack, ProgramHeap& heap)
        {
            Obj elem_ty, arr_ty;
            return
                stack.pop(elem_ty) ||
                heap.create_type_arr(arr_ty, elem_ty.type_obj) ||
                stack.push(arr_ty);
        }

        inline bool exec_aty(CallStack& stack, ProgramHeap& heap)
        {
            std::vector<TypeObj*> tys(oprand);
            for (size_t i = 0; i < oprand; i++)
            {
                Obj ty;
                if (stack.pop(ty))
                    return true;
                tys.push_back(ty.type_obj);
            }
            Obj type;
            return
                heap.create_type_agg(type, tys) ||
                stack.push(type) ||
                set_pc(pc + sizeof(size_t));
        }

        inline bool exec_pop(CallStack& stack)
        {
            return
                stack.del(oprand) ||
                set_pc(pc + sizeof(size_t));
        }

        inline bool exec_dup(CallStack& stack)
        {
            Obj obj;
            return
                stack.get(oprand, obj) ||
                stack.push(obj) ||
                set_pc(pc + sizeof(size_t));
        }

        inline bool exec_cpy(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, src, dst;
            return
                stack.pop(type) ||
                stack.pop(src) ||
                type.type_obj->cpy(heap, dst, src) ||
                stack.push(dst);
        }

        inline bool exec_inst(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, inst;
            return
                stack.pop(type) ||
                type.type_obj->inst(heap, inst) ||
                stack.push(inst);
        }

        inline bool exec_call(CallStack& stack)
        {
            Obj proc;
            return
                stack.pop(proc) ||
                push_pc() ||
                set_pc(proc.ptr);
        }

        inline bool exec_ret(CallStack& stack)
        {
            if (pc_stack.size() > 0)
                return pop_pc();

            complete = true;
            return false;
        }

        inline bool exec_set(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->set(heap, lhs, rhs);
        }

        inline bool exec_iadd(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->iadd(heap, lhs, rhs);
        }

        inline bool exec_isub(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->isub(heap, lhs, rhs);
        }

        inline bool exec_imul(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->imul(heap, lhs, rhs);
        }

        inline bool exec_idiv(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->idiv(heap, lhs, rhs);
        }

        inline bool exec_imod(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->imod(heap, lhs, rhs);
        }

        inline bool exec_add(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->add(heap, dst, lhs, rhs) ||
                stack.push(dst);
        }

        inline bool exec_sub(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->sub(heap, dst, lhs, rhs) ||
                stack.push(dst);
        }

        inline bool exec_mul(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->mul(heap, dst, lhs, rhs) ||
                stack.push(dst);
        }

        inline bool exec_div(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->div(heap, dst, lhs, rhs) ||
                stack.push(dst);
        }

        inline bool exec_mod(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->mod(heap, dst, lhs, rhs) ||
                stack.push(dst);
        }

        inline bool exec_eq(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->eq(heap, dst, lhs, rhs) ||
                stack.push(dst);
        }

        inline bool exec_ne(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->ne(heap, dst, lhs, rhs) ||
                stack.push(dst);
        }

        inline bool exec_gt(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->gt(heap, dst, lhs, rhs) ||
                stack.push(dst);
        }

        inline bool exec_lt(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->lt(heap, dst, lhs, rhs) ||
                stack.push(dst);
        }

        inline bool exec_ge(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->ge(heap, dst, lhs, rhs) ||
                stack.push(dst);
        }

        inline bool exec_le(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->le(heap, dst, lhs, rhs) ||
                stack.push(dst);
        }

        inline bool exec_idx(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->idx(heap, dst, lhs, rhs) ||
                stack.push(dst);
        }

        inline bool exec_xstr(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, src, dst;
            return
                stack.pop(type) ||
                stack.pop(src) ||
                type.type_obj->xstr(heap, dst, src) ||
                stack.push(dst);
        }

        inline bool exec_xflt(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, src, dst;
            return
                stack.pop(type) ||
                stack.pop(src) ||
                type.type_obj->xflt(heap, dst, src) ||
                stack.push(dst);
        }

        inline bool exec_xint(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, src, dst;
            return
                stack.pop(type) ||
                stack.pop(src) ||
                type.type_obj->xint(heap, dst, src) ||
                stack.push(dst);
        }

        inline bool exec_dsp(CallStack& stack)
        {
            Obj obj;
            if (stack.pop(obj))
                return true;
            std::cout << *obj.str_obj << std::endl;
            return false;
        }

        inline bool exec_err(CallStack& stack)
        {
            Obj obj;
            if (stack.pop(obj))
                return true;
            return error::runtime(*obj.str_obj);
        }

        inline bool exec_edg(CallStack& stack, ProgramHeap& heap)
        {
            Obj argnum;
            if (stack.pop(argnum))
                return true;
            std::vector<IntObj> dims{ *argnum.int_obj };
            for (IntObj i = 0; i < *argnum.int_obj; i++)
            {
                Obj dim;
                if (stack.pop(dim))
                    return true;
                dims.push_back(*dim.int_obj);
            }
            Obj obj, fty;
            return
                stack.pop(fty) ||
                heap.create_obj_edge(obj, *fty.fty_obj, dims) ||
                stack.push(obj);
        }

        inline bool exec_nde(CallStack& stack, ProgramHeap& heap)
        {
            Obj obj, name;
            return
                stack.pop(name) ||
                heap.create_obj_node(obj, *name.str_obj) ||
                stack.push(obj);
        }

        inline bool exec_ini(CallStack& stack, ProgramHeap& heap)
        {
            Obj obj, name;
            return
                stack.pop(name) ||
                heap.create_obj_init(obj, *name.str_obj) ||
                stack.push(obj);
        }

        inline bool exec_blk(CallStack& stack, ProgramHeap& heap)
        {
            Obj obj, name;
            return
                stack.pop(name) ||
                heap.create_obj_block(obj, *name.str_obj) ||
                stack.push(obj);
        }

        inline bool exec_tsr(CallStack& stack, ProgramHeap& heap)
        {
            Obj obj, forward, backward;
            return
                stack.pop(backward) ||
                stack.pop(forward) ||
                heap.create_obj_tensor(obj, forward.edge_obj, backward.edge_obj) ||
                stack.push(obj);
        }

        inline bool exec_fwd(CallStack& stack)
        {
            Obj obj, ten;
            return
                stack.pop(obj) ||
                pbuilder->get_forward(obj, ten.tensor_obj) ||
                stack.push(obj);
        }

        inline bool exec_bwd(CallStack& stack)
        {
            Obj obj, ten;
            return
                stack.pop(obj) ||
                pbuilder->get_backward(obj, ten.tensor_obj) ||
                stack.push(obj);
        }

        inline bool exec_ndcfg(CallStack& stack)
        {
            Obj node, name, type, obj;
            core::Config* cfg;
            return
                stack.pop(node) ||
                stack.pop(name) ||
                stack.pop(type) ||
                stack.pop(obj) ||
                type.type_obj->cfg(cfg, obj) ||
                pbuilder->add_ndcfg(*name.str_obj, node.node_obj, cfg) ||
                stack.push(node);
        }

        inline bool exec_bkcfg(CallStack& stack)
        {
            Obj block, name, type, obj;
            core::Config* cfg;
            return
                stack.pop(block) ||
                stack.pop(name) ||
                stack.pop(type) ||
                stack.pop(obj) ||
                type.type_obj->cfg(cfg, obj) ||
                pbuilder->add_bkcfg(*name.str_obj, block.block_obj, cfg) ||
                stack.push(block);
        }

        inline bool exec_incfg(CallStack& stack)
        {
            Obj init, name, type, obj;
            core::Config* cfg;
            return
                stack.pop(init) ||
                stack.pop(name) ||
                stack.pop(type) ||
                stack.pop(obj) ||
                type.type_obj->cfg(cfg, obj) ||
                pbuilder->add_incfg(*name.str_obj, init.init_obj, cfg) ||
                stack.push(init);
        }

        inline bool exec_ndinp(CallStack& stack)
        {
            Obj name, node, edge;
            return
                stack.pop(node) ||
                stack.pop(name) ||
                stack.pop(edge) ||
                pbuilder->set_ndinp(*name.str_obj, node.node_obj, edge.edge_obj) ||
                stack.push(node);
        }

        inline bool exec_ndout(CallStack& stack)
        {
            Obj name, node, edge;
            return
                stack.pop(node) ||
                stack.pop(name) ||
                stack.pop(edge) ||
                pbuilder->set_ndout(*name.str_obj, node.node_obj, edge.edge_obj) ||
                stack.push(node);
        }

        inline bool exec_bkprt(CallStack& stack)
        {
            Obj block, parent;
            return
                stack.pop(block) ||
                stack.pop(parent) ||
                pbuilder->set_bkprt(block.block_obj, parent.block_obj) ||
                stack.push(block);
        }

        inline bool exec_bkinp(CallStack& stack)
        {
            Obj name, block, tensor;
            return
                stack.pop(block) ||
                stack.pop(name) ||
                stack.pop(tensor) ||
                pbuilder->set_bkinp(*name.str_obj, block.block_obj, tensor.tensor_obj) ||
                stack.push(block);
        }

        inline bool exec_bkout(CallStack& stack)
        {
            Obj name, block, tensor;
            return
                stack.pop(block) ||
                stack.pop(name) ||
                stack.pop(tensor) ||
                pbuilder->set_bkout(*name.str_obj, block.block_obj, tensor.tensor_obj) ||
                stack.push(block);
        }

        inline bool exec_pshmd(CallStack& stack)
        {
            Obj mode;
            return
                stack.pop(mode) ||
                push_md(*mode.str_obj);
        }

        inline bool exec_popmd(CallStack& stack)
        {
            return
                pop_md();
        }

        inline bool exec_ext(CallStack& stack)
        {
            Obj name, block, tensor, init;
            return
                stack.pop(block) ||
                stack.pop(init) ||
                stack.pop(name) ||
                stack.pop(tensor) ||
                pbuilder->add_extern(*name.str_obj, block.block_obj, tensor.tensor_obj, init.init_obj);
        }

        inline bool exec_exp(CallStack& stack)
        {
            Obj name, tensor;
            return
                stack.pop(name) ||
                stack.pop(tensor) ||
                pbuilder->add_export(*name.str_obj, tensor.tensor_obj);
        }

        bool exec(CallStack& stack, ProgramHeap& heap, ByteCode& byte_code, std::string entry_point, core::Graph& graph)
		{
            if (!byte_code.proc_offsets.contains(entry_point))
                return error::runtime("Unable to find entry point '%'", entry_point);

            // Initializing the interpreter state
            pc = byte_code.proc_offsets.at(entry_point);
            code = byte_code.code_segment;
            data = byte_code.data_segment;
            complete = false;
            pc_stack.clear();
            GraphBuilder builder;
            pbuilder = &builder;
            md_stack.clear();
            error::bind_runtime_context(byte_code.debug_info, pc);
            if (stack.push(Obj{ .block_obj = nullptr }))  // this will mark the first block as root
                return true;

            InstructionType ty;
            while (!complete)
            {
                ty = *(InstructionType*)(code + pc);
                pc += sizeof(InstructionType);
                switch (ty)
                {
                case InstructionType::JMP:
                    if (exec_jmp())
                        goto runtime_error;
                    break;
                case InstructionType::BRT:
                    if (exec_brt(stack))
                        goto runtime_error;
                    break;
                case InstructionType::BRF:
                    if (exec_brf(stack))
                        goto runtime_error;
                    break;
                case InstructionType::NEW:
                    if (exec_new(stack))
                        goto runtime_error;
                    break;
                case InstructionType::AGG:
                    if (exec_agg(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::ARR:
                    if (exec_arr(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::ATY:
                    if (exec_aty(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::POP:
                    if (exec_pop(stack))
                        goto runtime_error;
                    break;
                case InstructionType::DUP:
                    if (exec_dup(stack))
                        goto runtime_error;
                    break;
                case InstructionType::CPY:
                    if (exec_cpy(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::INST:
                    if (exec_inst(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::CALL:
                    if (exec_call(stack))
                        goto runtime_error;
                    break;
                case InstructionType::RET:
                    if (exec_ret(stack))
                        goto runtime_error;
                    break;
                case InstructionType::SET:
                    if (exec_set(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::IADD:
                    if (exec_iadd(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::ISUB:
                    if (exec_isub(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::IMUL:
                    if (exec_imul(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::IDIV:
                    if (exec_idiv(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::IMOD:
                    if (exec_imod(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::ADD:
                    if (exec_add(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::SUB:
                    if (exec_sub(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::MUL:
                    if (exec_mul(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::DIV:
                    if (exec_div(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::MOD:
                    if (exec_mod(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::EQ:
                    if (exec_eq(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::NE:
                    if (exec_ne(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::GT:
                    if (exec_gt(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::LT:
                    if (exec_lt(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::GE:
                    if (exec_ge(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::LE:
                    if (exec_le(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::IDX:
                    if (exec_idx(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::XSTR:
                    if (exec_xstr(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::XFLT:
                    if (exec_xflt(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::XINT:
                    if (exec_xint(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::DSP:
                    if (exec_dsp(stack))
                        goto runtime_error;
                    break;
                case InstructionType::ERR:
                    if (exec_err(stack))
                        goto runtime_error;
                    break;

                case InstructionType::EDG:
                    if (exec_edg(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::NDE:
                    if (exec_nde(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::INI:
                    if (exec_ini(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::BLK:
                    if (exec_blk(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::TSR:
                    if (exec_tsr(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::FWD:
                    if (exec_fwd(stack))
                        goto runtime_error;
                    break;
                case InstructionType::BWD:
                    if (exec_bwd(stack))
                        goto runtime_error;
                    break;
                case InstructionType::NDCFG:
                    if (exec_ndcfg(stack))
                        goto runtime_error;
                    break;
                case InstructionType::BKCFG:
                    if (exec_bkcfg(stack))
                        goto runtime_error;
                    break;
                case InstructionType::INCFG:
                    if (exec_incfg(stack))
                        goto runtime_error;
                    break;
                case InstructionType::NDINP:
                    if (exec_ndinp(stack))
                        goto runtime_error;
                    break;
                case InstructionType::NDOUT:
                    if (exec_ndout(stack))
                        goto runtime_error;
                    break;
                case InstructionType::BKPRT:
                    if (exec_bkprt(stack))
                        goto runtime_error;
                    break;
                case InstructionType::BKINP:
                    if (exec_bkinp(stack))
                        goto runtime_error;
                    break;
                case InstructionType::BKOUT:
                    if (exec_bkout(stack))
                        goto runtime_error;
                    break;
                case InstructionType::PSHMD:
                    if (exec_pshmd(stack))
                        goto runtime_error;
                    break;
                case InstructionType::POPMD:
                    if (exec_popmd(stack))
                        goto runtime_error;
                    break;
                case InstructionType::EXT:
                    if (exec_ext(stack))
                        goto runtime_error;
                    break;
                case InstructionType::EXP:
                    if (exec_exp(stack))
                        goto runtime_error;
                    break;

                default:
                    error::runtime("Invalid instruction opcode '%'", (uint8_t)ty);
                    goto runtime_error;
                }
            }
            return pbuilder->export_graph(graph);

        runtime_error:
            // TODO: unwind the call stack with location info
            return true;
		}
	}
}
