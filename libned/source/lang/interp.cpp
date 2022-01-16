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

        bool GraphBuilder::create_edge(core::Edge* pedge)
        {
            pedge = new core::Edge();
            edge_buffer.push_back(pedge);
            return false;
        }

        bool GraphBuilder::create_node(const std::string& name, core::Node* pnode)
        {
            pnode = new core::Node();
            pnode->name = name;
            node_buffer.push_back(pnode);
            return false;
        }

        bool GraphBuilder::create_block(const std::string& name, core::Block* pblock)
        {
            pblock = new core::Block();
            pblock->name = name;
            block_buffer.push_back(pblock);
            return false;
        }

        bool GraphBuilder::set_child(core::Block* pparent, core::Block* pchild)
        {
            return error::runtime("GraphBuilder::set_child has not been implemented");
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

        inline bool exec_edg(CallStack& stack)
        {
            Obj edge{};
            return
                pbuilder->create_edge(edge.edge_obj) ||
                stack.push(edge);
        }

        inline bool exec_nde(CallStack& stack)
        {
            Obj name, node{};
            return
                stack.pop(name) ||
                pbuilder->create_node(*name.str_obj, node.node_obj) ||
                stack.push(node);
        }

        inline bool exec_blk(CallStack& stack)
        {
            Obj name, block{};
            return
                stack.pop(name) ||
                pbuilder->create_block(*name.str_obj, block.block_obj) ||
                stack.push(block);
        }

        inline bool exec_bksub(CallStack& stack)
        {
            Obj child, parent;
            return
                stack.pop(child) ||
                stack.pop(parent) ||
                pbuilder->set_child(parent.block_obj, child.block_obj);
        }

        inline bool exec_ndinp(CallStack& stack)
        {
            Obj edge, name, node;
            return
                stack.pop(edge) ||
                stack.pop(name) ||
                stack.pop(node) ||
                pbuilder->set_ndinp(node.node_obj, edge.edge_obj, *name.str_obj);
        }

        inline bool exec_ndout(CallStack& stack)
        {
            Obj edge, name, node;
            return
                stack.pop(edge) ||
                stack.pop(name) ||
                stack.pop(node) ||
                pbuilder->set_ndout(node.node_obj, edge.edge_obj, *name.str_obj);
        }

        inline bool exec_bkinp(CallStack& stack)
        {
            Obj forward, backward, name, block;
            return
                stack.pop(forward) ||
                stack.pop(backward) ||
                stack.pop(name) ||
                stack.pop(block) ||
                pbuilder->set_bkinp(block.block_obj, forward.edge_obj, backward.edge_obj, *name.str_obj);
        }

        inline bool exec_bkout(CallStack& stack)
        {
            Obj forward, backward, name, block;
            return
                stack.pop(forward) ||
                stack.pop(backward) ||
                stack.pop(name) ||
                stack.pop(block) ||
                pbuilder->set_bkout(block.block_obj, forward.edge_obj, backward.edge_obj, *name.str_obj);
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
            Obj forward, backward, name, block;
            return
                stack.pop(forward) ||
                stack.pop(backward) ||
                stack.pop(name) ||
                stack.pop(block) ||
                pbuilder->set_weight(block.block_obj, forward.edge_obj, backward.edge_obj, *name.str_obj);
        }

        inline bool exec_exp(CallStack& stack)
        {
            Obj forward, backward, name;
            return
                stack.pop(forward) ||
                stack.pop(backward) ||
                stack.pop(name) ||
                pbuilder->set_export(forward.edge_obj, backward.edge_obj, *name.str_obj);
        }

        bool exec(CallStack& stack, ProgramHeap& heap, ByteCode& byte_code, std::string entry_point, core::Graph& graph)
		{
            if (!byte_code.proc_offsets.contains(entry_point))
                return error::runtime("", 0ULL, 0ULL, "Unable to find entry point '%'", entry_point);

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

            // TODO: setup the stack with the input edges

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

                case InstructionType::EDG:
                    if (exec_edg(stack))
                        goto runtime_error;
                    break;
                case InstructionType::NDE:
                    if (exec_nde(stack))
                        goto runtime_error;
                    break;
                case InstructionType::BLK:
                    if (exec_blk(stack))
                        goto runtime_error;
                    break;
                case InstructionType::BKSUB:
                    if (exec_bksub(stack))
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
			return false;

        runtime_error:
            // TODO: unwind the call stack with location info
            return true;
		}
	}
}
