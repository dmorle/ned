#include <ned/lang/interp.h>
#include <ned/lang/bytecode.h>

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

        bool GraphBuilder::create_edge(RuntimeErrors& errs, core::Edge* pedge)
        {
            pedge = new core::Edge();
            edge_buffer.push_back(pedge);
            return false;
        }

        bool GraphBuilder::create_node(RuntimeErrors& errs, const std::string& name, core::Node* pnode)
        {
            pnode = new core::Node();
            pnode->name = name;
            node_buffer.push_back(pnode);
            return false;
        }

        bool GraphBuilder::create_block(RuntimeErrors& errs, const std::string& name, core::Block* pblock)
        {
            pblock = new core::Block();
            pblock->name = name;
            block_buffer.push_back(pblock);
            return false;
        }

        bool GraphBuilder::set_child(RuntimeErrors& errs, core::Block* pparent, core::Block* pchild)
        {
            return errs.add("GraphBuilder::set_child has not been implemented");
        }

        // stack operations

        bool CallStack::pop(RuntimeErrors& errs, Obj& obj)
        {
            if (sp == 0)
                return errs.add("Stack pointer out of bounds during pop operation");
            obj = stack[--sp];
            return false;
        }

        bool CallStack::del(RuntimeErrors& errs, size_t i)
        {
            if (i >= sp)
                return errs.add("Attempted to delete a non-existent stack element");
            sp--;
            for (size_t j = sp - i; j < sp; j++)
                stack[j] = stack[j + 1];
            return false;
        }

        bool CallStack::get(RuntimeErrors& errs, size_t i, Obj& obj)
        {
            if (i >= sp)
                return errs.add("Attempted to retrieve a non-existent stack element");
            obj = stack[sp - i - 1];
            return false;
        }

        bool CallStack::push(RuntimeErrors& errs, Obj obj)
        {
            if (sp >= stack.size())
                return errs.add("Stack overflow error");
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

        inline bool push_md(RuntimeErrors& errs, const std::string& mode)
        {
            md_stack.push_back(mode);
            return false;
        }

        inline bool pop_md(RuntimeErrors& errs)
        {
            if (md_stack.size() == 0)
                return errs.add("Attempted to release a non-existent evaluation mode");
            md_stack.pop_back();
            return false;
        }

        // instruction implementations

        inline bool exec_jmp(RuntimeErrors& errs)
        {
            return
                set_pc(oprand);
        }

        inline bool exec_brt(RuntimeErrors& errs, CallStack& stack)
        {
            Obj obj;
            return
                stack.pop(errs, obj) ||
                set_pc(*obj.bool_obj ? oprand : pc + sizeof(size_t));
        }

        inline bool exec_brf(RuntimeErrors& errs, CallStack& stack)
        {
            Obj obj;
            return
                stack.pop(errs, obj) ||
                set_pc(*obj.bool_obj ? pc + sizeof(size_t) : oprand);
        }

        inline bool exec_new(RuntimeErrors& errs, CallStack& stack)
        {
            return
                stack.push(errs, data[oprand]) ||
                set_pc(pc + sizeof(size_t));
        }

        inline bool exec_agg(RuntimeErrors& errs, CallStack& stack, ProgramHeap& heap)
        {
            std::vector<Obj> objs(oprand);
            for (size_t i = 0; i < oprand; i++)
            {
                Obj obj;
                if (stack.pop(errs, obj))
                    return true;
                objs.push_back(obj);
            }
            Obj dst;
            return
                heap.create_obj_agg(errs, dst, objs) ||
                stack.push(errs, dst) ||
                set_pc(pc + sizeof(size_t));
        }

        inline bool exec_arr(RuntimeErrors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj elem_ty, arr_ty;
            return
                stack.pop(errs, elem_ty) ||
                heap.create_type_arr(errs, arr_ty, elem_ty.type_obj) ||
                stack.push(errs, arr_ty);
        }

        inline bool exec_aty(RuntimeErrors& errs, CallStack& stack, ProgramHeap& heap)
        {
            std::vector<TypeObj*> tys(oprand);
            for (size_t i = 0; i < oprand; i++)
            {
                Obj ty;
                if (stack.pop(errs, ty))
                    return true;
                tys.push_back(ty.type_obj);
            }
            Obj type;
            return
                heap.create_type_agg(errs, type, tys) ||
                stack.push(errs, type) ||
                set_pc(pc + sizeof(size_t));
        }

        inline bool exec_pop(RuntimeErrors& errs, CallStack& stack)
        {
            return
                stack.del(errs, oprand) ||
                set_pc(pc + sizeof(size_t));
        }

        inline bool exec_dup(RuntimeErrors& errs, CallStack& stack)
        {
            Obj obj;
            return
                stack.get(errs, oprand, obj) ||
                stack.push(errs, obj) ||
                set_pc(pc + sizeof(size_t));
        }

        inline bool exec_cpy(RuntimeErrors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, src, dst;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, src) ||
                type.type_obj->cpy(errs, heap, dst, src) ||
                stack.push(errs, dst);
        }

        inline bool exec_inst(RuntimeErrors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, inst;
            return
                stack.pop(errs, type) ||
                type.type_obj->inst(errs, heap, inst) ||
                stack.push(errs, inst);
        }

        inline bool exec_call(RuntimeErrors& errs, CallStack& stack)
        {
            Obj proc;
            return
                stack.pop(errs, proc) ||
                push_pc() ||
                set_pc(proc.ptr);
        }

        inline bool exec_ret(RuntimeErrors& errs, CallStack& stack)
        {
            if (pc_stack.size() > 0)
                return pop_pc();

            complete = true;
            return false;
        }

        inline bool exec_set(RuntimeErrors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, rhs) ||
                stack.pop(errs, lhs) ||
                type.type_obj->set(errs, heap, lhs, rhs);
        }

        inline bool exec_iadd(RuntimeErrors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, rhs) ||
                stack.pop(errs, lhs) ||
                type.type_obj->iadd(errs, heap, lhs, rhs);
        }

        inline bool exec_isub(RuntimeErrors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, rhs) ||
                stack.pop(errs, lhs) ||
                type.type_obj->isub(errs, heap, lhs, rhs);
        }

        inline bool exec_imul(RuntimeErrors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, rhs) ||
                stack.pop(errs, lhs) ||
                type.type_obj->imul(errs, heap, lhs, rhs);
        }

        inline bool exec_idiv(RuntimeErrors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, rhs) ||
                stack.pop(errs, lhs) ||
                type.type_obj->idiv(errs, heap, lhs, rhs);
        }

        inline bool exec_imod(RuntimeErrors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, rhs) ||
                stack.pop(errs, lhs) ||
                type.type_obj->imod(errs, heap, lhs, rhs);
        }

        inline bool exec_add(RuntimeErrors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, rhs) ||
                stack.pop(errs, lhs) ||
                type.type_obj->add(errs, heap, dst, lhs, rhs) ||
                stack.push(errs, dst);
        }

        inline bool exec_sub(RuntimeErrors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, rhs) ||
                stack.pop(errs, lhs) ||
                type.type_obj->sub(errs, heap, dst, lhs, rhs) ||
                stack.push(errs, dst);
        }

        inline bool exec_mul(RuntimeErrors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, rhs) ||
                stack.pop(errs, lhs) ||
                type.type_obj->mul(errs, heap, dst, lhs, rhs) ||
                stack.push(errs, dst);
        }

        inline bool exec_div(RuntimeErrors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, rhs) ||
                stack.pop(errs, lhs) ||
                type.type_obj->div(errs, heap, dst, lhs, rhs) ||
                stack.push(errs, dst);
        }

        inline bool exec_mod(RuntimeErrors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, rhs) ||
                stack.pop(errs, lhs) ||
                type.type_obj->mod(errs, heap, dst, lhs, rhs) ||
                stack.push(errs, dst);
        }

        inline bool exec_eq(RuntimeErrors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, rhs) ||
                stack.pop(errs, lhs) ||
                type.type_obj->eq(errs, heap, dst, lhs, rhs) ||
                stack.push(errs, dst);
        }

        inline bool exec_ne(RuntimeErrors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, rhs) ||
                stack.pop(errs, lhs) ||
                type.type_obj->ne(errs, heap, dst, lhs, rhs) ||
                stack.push(errs, dst);
        }

        inline bool exec_gt(RuntimeErrors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, rhs) ||
                stack.pop(errs, lhs) ||
                type.type_obj->gt(errs, heap, dst, lhs, rhs) ||
                stack.push(errs, dst);
        }

        inline bool exec_lt(RuntimeErrors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, rhs) ||
                stack.pop(errs, lhs) ||
                type.type_obj->lt(errs, heap, dst, lhs, rhs) ||
                stack.push(errs, dst);
        }

        inline bool exec_ge(RuntimeErrors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, rhs) ||
                stack.pop(errs, lhs) ||
                type.type_obj->ge(errs, heap, dst, lhs, rhs) ||
                stack.push(errs, dst);
        }

        inline bool exec_le(RuntimeErrors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, rhs) ||
                stack.pop(errs, lhs) ||
                type.type_obj->le(errs, heap, dst, lhs, rhs) ||
                stack.push(errs, dst);
        }

        inline bool exec_idx(RuntimeErrors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, rhs) ||
                stack.pop(errs, lhs) ||
                type.type_obj->idx(errs, heap, dst, lhs, rhs) ||
                stack.push(errs, dst);
        }

        inline bool exec_xstr(RuntimeErrors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, src, dst;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, src) ||
                type.type_obj->xstr(errs, heap, dst, src) ||
                stack.push(errs, dst);
        }

        inline bool exec_xflt(RuntimeErrors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, src, dst;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, src) ||
                type.type_obj->xflt(errs, heap, dst, src) ||
                stack.push(errs, dst);
        }

        inline bool exec_xint(RuntimeErrors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, src, dst;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, src) ||
                type.type_obj->xint(errs, heap, dst, src) ||
                stack.push(errs, dst);
        }

        inline bool exec_dsp(RuntimeErrors& errs, CallStack& stack)
        {
            Obj obj;
            if (stack.pop(errs, obj))
                return true;
            std::cout << *obj.str_obj << std::endl;
            return false;
        }

        inline bool exec_edg(RuntimeErrors& errs, CallStack& stack)
        {
            Obj edge;
            return
                pbuilder->create_edge(errs, edge.edge_obj) ||
                stack.push(errs, edge);
        }

        inline bool exec_nde(RuntimeErrors& errs, CallStack& stack)
        {
            Obj name, node;
            return
                stack.pop(errs, name) ||
                pbuilder->create_node(errs, *name.str_obj, node.node_obj) ||
                stack.push(errs, node);
        }

        inline bool exec_blk(RuntimeErrors& errs, CallStack& stack)
        {
            Obj name, block;
            return
                stack.pop(errs, name) ||
                pbuilder->create_block(errs, *name.str_obj, block.block_obj) ||
                stack.push(errs, block);
        }

        inline bool exec_bksub(RuntimeErrors& errs, CallStack& stack)
        {
            Obj child, parent;
            return
                stack.pop(errs, child) ||
                stack.pop(errs, parent) ||
                pbuilder->set_child(errs, parent.block_obj, child.block_obj);
        }

        inline bool exec_ndinp(RuntimeErrors& errs, CallStack& stack)
        {
            Obj edge, name, node;
            return
                stack.pop(errs, edge) ||
                stack.pop(errs, name) ||
                stack.pop(errs, node) ||
                pbuilder->set_ndinp(errs, node.node_obj, edge.edge_obj, *name.str_obj);
        }

        inline bool exec_ndout(RuntimeErrors& errs, CallStack& stack)
        {
            Obj edge, name, node;
            return
                stack.pop(errs, edge) ||
                stack.pop(errs, name) ||
                stack.pop(errs, node) ||
                pbuilder->set_ndout(errs, node.node_obj, edge.edge_obj, *name.str_obj);
        }

        inline bool exec_bkinp(RuntimeErrors& errs, CallStack& stack)
        {
            Obj forward, backward, name, block;
            return
                stack.pop(errs, forward) ||
                stack.pop(errs, backward) ||
                stack.pop(errs, name) ||
                stack.pop(errs, block) ||
                pbuilder->set_bkinp(errs, block.block_obj, forward.edge_obj, backward.edge_obj, *name.str_obj);
        }

        inline bool exec_bkout(RuntimeErrors& errs, CallStack& stack)
        {
            Obj forward, backward, name, block;
            return
                stack.pop(errs, forward) ||
                stack.pop(errs, backward) ||
                stack.pop(errs, name) ||
                stack.pop(errs, block) ||
                pbuilder->set_bkout(errs, block.block_obj, forward.edge_obj, backward.edge_obj, *name.str_obj);
        }

        inline bool exec_pshmd(RuntimeErrors& errs, CallStack& stack)
        {
            Obj mode;
            return
                stack.pop(errs, mode) ||
                push_md(errs, *mode.str_obj);
        }

        inline bool exec_popmd(RuntimeErrors& errs, CallStack& stack)
        {
            return
                pop_md(errs);
        }

        inline bool exec_ext(RuntimeErrors& errs, CallStack& stack)
        {
            Obj forward, backward, name, block;
            return
                stack.pop(errs, forward) ||
                stack.pop(errs, backward) ||
                stack.pop(errs, name) ||
                stack.pop(errs, block) ||
                pbuilder->set_weight(errs, block.block_obj, forward.edge_obj, backward.edge_obj, *name.str_obj);
        }

        inline bool exec_exp(RuntimeErrors& errs, CallStack& stack)
        {
            Obj forward, backward, name;
            return
                stack.pop(errs, forward) ||
                stack.pop(errs, backward) ||
                stack.pop(errs, name) ||
                pbuilder->set_export(errs, forward.edge_obj, backward.edge_obj, *name.str_obj);
        }

        bool exec(Errors& errs, CallStack& stack, ProgramHeap& heap, ByteCode& byte_code, std::string entry_point, core::Graph& graph)
		{
            if (!byte_code.proc_offsets.contains(entry_point))
                return errs.add("", 0ULL, 0ULL, "Unable to find entry point '{}'", entry_point);

            // Initializing the interpreter state
            pc = byte_code.proc_offsets.at(entry_point);
            code = byte_code.code_segment;
            data = byte_code.data_segment;
            complete = false;
            pc_stack.clear();

            GraphBuilder builder;
            pbuilder = &builder;
            md_stack.clear();

            RuntimeErrors rt_errs{ errs, byte_code.debug_info, pc };

            // TODO: setup the stack with the input edges

            InstructionType ty;
            while (!complete)
            {
                ty = *(InstructionType*)(code + pc);
                pc += sizeof(InstructionType);
                switch (ty)
                {
                case InstructionType::JMP:
                    if (exec_jmp(rt_errs))
                        goto runtime_error;
                    break;
                case InstructionType::BRT:
                    if (exec_brt(rt_errs, stack))
                        goto runtime_error;
                    break;
                case InstructionType::BRF:
                    if (exec_brf(rt_errs, stack))
                        goto runtime_error;
                    break;
                case InstructionType::NEW:
                    if (exec_new(rt_errs, stack))
                        goto runtime_error;
                    break;
                case InstructionType::AGG:
                    if (exec_agg(rt_errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::ARR:
                    if (exec_arr(rt_errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::ATY:
                    if (exec_aty(rt_errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::POP:
                    if (exec_pop(rt_errs, stack))
                        goto runtime_error;
                    break;
                case InstructionType::DUP:
                    if (exec_dup(rt_errs, stack))
                        goto runtime_error;
                    break;
                case InstructionType::CPY:
                    if (exec_cpy(rt_errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::INST:
                    if (exec_inst(rt_errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::CALL:
                    if (exec_call(rt_errs, stack))
                        goto runtime_error;
                    break;
                case InstructionType::RET:
                    if (exec_ret(rt_errs, stack))
                        goto runtime_error;
                    break;
                case InstructionType::SET:
                    if (exec_set(rt_errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::IADD:
                    if (exec_iadd(rt_errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::ISUB:
                    if (exec_isub(rt_errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::IMUL:
                    if (exec_imul(rt_errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::IDIV:
                    if (exec_idiv(rt_errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::IMOD:
                    if (exec_imod(rt_errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::ADD:
                    if (exec_add(rt_errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::SUB:
                    if (exec_sub(rt_errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::MUL:
                    if (exec_mul(rt_errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::DIV:
                    if (exec_div(rt_errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::MOD:
                    if (exec_mod(rt_errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::EQ:
                    if (exec_eq(rt_errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::NE:
                    if (exec_ne(rt_errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::GT:
                    if (exec_gt(rt_errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::LT:
                    if (exec_lt(rt_errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::GE:
                    if (exec_ge(rt_errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::LE:
                    if (exec_le(rt_errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::IDX:
                    if (exec_idx(rt_errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::XSTR:
                    if (exec_xstr(rt_errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::XFLT:
                    if (exec_xflt(rt_errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::XINT:
                    if (exec_xint(rt_errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::DSP:
                    if (exec_dsp(rt_errs, stack))
                        goto runtime_error;
                    break;

                case InstructionType::EDG:
                    if (exec_edg(rt_errs, stack))
                        goto runtime_error;
                    break;
                case InstructionType::NDE:
                    if (exec_nde(rt_errs, stack))
                        goto runtime_error;
                    break;
                case InstructionType::BLK:
                    if (exec_blk(rt_errs, stack))
                        goto runtime_error;
                    break;
                case InstructionType::BKSUB:
                    if (exec_bksub(rt_errs, stack))
                        goto runtime_error;
                    break;
                case InstructionType::NDINP:
                    if (exec_ndinp(rt_errs, stack))
                        goto runtime_error;
                    break;
                case InstructionType::NDOUT:
                    if (exec_ndout(rt_errs, stack))
                        goto runtime_error;
                    break;
                case InstructionType::BKINP:
                    if (exec_bkinp(rt_errs, stack))
                        goto runtime_error;
                    break;
                case InstructionType::BKOUT:
                    if (exec_bkout(rt_errs, stack))
                        goto runtime_error;
                    break;
                case InstructionType::PSHMD:
                    if (exec_pshmd(rt_errs, stack))
                        goto runtime_error;
                    break;
                case InstructionType::POPMD:
                    if (exec_popmd(rt_errs, stack))
                        goto runtime_error;
                    break;
                case InstructionType::EXT:
                    if (exec_ext(rt_errs, stack))
                        goto runtime_error;
                    break;
                case InstructionType::EXP:
                    if (exec_exp(rt_errs, stack))
                        goto runtime_error;
                    break;

                default:
                    rt_errs.add("Invalid instruction opcode '{}'", ty);
                    goto runtime_error;
                }
            }
			return false;

        runtime_error:
            // TODO: unwind the call stack with location info
            return true;
		}
	}
}
