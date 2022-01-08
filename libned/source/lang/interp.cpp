#include <ned/lang/interp.h>
#include <ned/lang/bytecode.h>

#define oprand (*((size_t*)(code + pc)))

namespace nn
{
	namespace lang
	{
        // stack operations
        // TODO: figure out runtime errors
        
        bool CallStack::pop(Errors& errs, Obj& obj)
        {
            assert(sp != 0);
            obj = stack[--sp];
            return false;
        }

        bool CallStack::del(Errors& errs, size_t i)
        {
            assert(i < sp);
            sp--;
            for (size_t j = sp - i; j < sp; j++)
                stack[j] = stack[j + 1];
            return false;
        }

        bool CallStack::get(Errors& errs, size_t i, Obj& obj)
        {
            assert(i < sp);
            obj = stack[sp - i - 1];
            return false;
        }

        bool CallStack::push(Errors& errs, Obj obj)
        {
            assert(sp < stack.size());
            stack[sp++] = obj;
            return false;
        }

        // state of the interpreter
        size_t pc = 0;
        CodeSegPtr code;
        DataSegPtr data;
        bool complete;
        std::vector<size_t> call_stack;

        // helper functions

        inline bool set_pc(size_t val)
        {
            pc = val;
            return false;
        }

        inline bool push_pc()
        {
            call_stack.push_back(pc);
            return false;
        }

        inline bool pop_pc()
        {
            assert(call_stack.size() > 0);
            pc = call_stack.back();
            call_stack.pop_back();
            return false;
        }

        // instruction implementations

        inline bool exec_jmp(Errors& errs)
        {
            return
                set_pc(oprand);
        }

        inline bool exec_brt(Errors& errs, CallStack& stack)
        {
            Obj obj;
            return
                stack.pop(errs, obj) ||
                set_pc(*obj.bool_obj ? oprand : pc + sizeof(size_t));
        }

        inline bool exec_brf(Errors& errs, CallStack& stack)
        {
            Obj obj;
            return
                stack.pop(errs, obj) ||
                set_pc(*obj.bool_obj ? pc + sizeof(size_t) : oprand);
        }

        inline bool exec_new(Errors& errs, CallStack& stack)
        {
            return
                stack.push(errs, data[oprand]) ||
                set_pc(pc + sizeof(size_t));
        }

        inline bool exec_agg(Errors& errs, CallStack& stack, ProgramHeap& heap)
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

        inline bool exec_arr(Errors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj elem_ty, arr_ty;
            return
                stack.pop(errs, elem_ty) ||
                heap.create_type_arr(errs, arr_ty, elem_ty.type_obj) ||
                stack.push(errs, arr_ty);
        }

        inline bool exec_aty(Errors& errs, CallStack& stack, ProgramHeap& heap)
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

        inline bool exec_pop(Errors& errs, CallStack& stack)
        {
            return
                stack.del(errs, oprand) ||
                set_pc(pc + sizeof(size_t));
        }

        inline bool exec_dup(Errors& errs, CallStack& stack)
        {
            Obj obj;
            return
                stack.get(errs, oprand, obj) ||
                stack.push(errs, obj) ||
                set_pc(pc + sizeof(size_t));
        }

        inline bool exec_cpy(Errors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, src, dst;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, src) ||
                type.type_obj->cpy(errs, heap, dst, src) ||
                stack.push(errs, dst);
        }

        inline bool exec_inst(Errors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, inst;
            return
                stack.pop(errs, type) ||
                type.type_obj->inst(errs, heap, inst) ||
                stack.push(errs, inst);
        }

        inline bool exec_call(Errors& errs, CallStack& stack)
        {
            Obj proc;
            return
                stack.pop(errs, proc) ||
                set_pc(pc + sizeof(size_t)) ||
                push_pc() ||
                set_pc(proc.ptr);
        }

        inline bool exec_ret(Errors& errs, CallStack& stack)
        {
            if (call_stack.size() > 0)
                return pop_pc();

            complete = true;
            return false;
        }

        inline bool exec_set(Errors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, rhs) ||
                stack.pop(errs, lhs) ||
                type.type_obj->set(errs, heap, lhs, rhs);
        }

        inline bool exec_iadd(Errors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, rhs) ||
                stack.pop(errs, lhs) ||
                type.type_obj->iadd(errs, heap, lhs, rhs);
        }

        inline bool exec_isub(Errors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, rhs) ||
                stack.pop(errs, lhs) ||
                type.type_obj->isub(errs, heap, lhs, rhs);
        }

        inline bool exec_imul(Errors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, rhs) ||
                stack.pop(errs, lhs) ||
                type.type_obj->imul(errs, heap, lhs, rhs);
        }

        inline bool exec_idiv(Errors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, rhs) ||
                stack.pop(errs, lhs) ||
                type.type_obj->idiv(errs, heap, lhs, rhs);
        }

        inline bool exec_imod(Errors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, rhs) ||
                stack.pop(errs, lhs) ||
                type.type_obj->imod(errs, heap, lhs, rhs);
        }

        inline bool exec_add(Errors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, rhs) ||
                stack.pop(errs, lhs) ||
                type.type_obj->add(errs, heap, dst, lhs, rhs) ||
                stack.push(errs, dst);
        }

        inline bool exec_sub(Errors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, rhs) ||
                stack.pop(errs, lhs) ||
                type.type_obj->sub(errs, heap, dst, lhs, rhs) ||
                stack.push(errs, dst);
        }

        inline bool exec_mul(Errors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, rhs) ||
                stack.pop(errs, lhs) ||
                type.type_obj->mul(errs, heap, dst, lhs, rhs) ||
                stack.push(errs, dst);
        }

        inline bool exec_div(Errors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, rhs) ||
                stack.pop(errs, lhs) ||
                type.type_obj->div(errs, heap, dst, lhs, rhs) ||
                stack.push(errs, dst);
        }

        inline bool exec_mod(Errors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, rhs) ||
                stack.pop(errs, lhs) ||
                type.type_obj->mod(errs, heap, dst, lhs, rhs) ||
                stack.push(errs, dst);
        }

        inline bool exec_eq(Errors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, rhs) ||
                stack.pop(errs, lhs) ||
                type.type_obj->eq(errs, heap, dst, lhs, rhs) ||
                stack.push(errs, dst);
        }

        inline bool exec_ne(Errors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, rhs) ||
                stack.pop(errs, lhs) ||
                type.type_obj->ne(errs, heap, dst, lhs, rhs) ||
                stack.push(errs, dst);
        }

        inline bool exec_gt(Errors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, rhs) ||
                stack.pop(errs, lhs) ||
                type.type_obj->gt(errs, heap, dst, lhs, rhs) ||
                stack.push(errs, dst);
        }

        inline bool exec_lt(Errors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, rhs) ||
                stack.pop(errs, lhs) ||
                type.type_obj->lt(errs, heap, dst, lhs, rhs) ||
                stack.push(errs, dst);
        }

        inline bool exec_ge(Errors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, rhs) ||
                stack.pop(errs, lhs) ||
                type.type_obj->ge(errs, heap, dst, lhs, rhs) ||
                stack.push(errs, dst);
        }

        inline bool exec_le(Errors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, rhs) ||
                stack.pop(errs, lhs) ||
                type.type_obj->le(errs, heap, dst, lhs, rhs) ||
                stack.push(errs, dst);
        }

        inline bool exec_idx(Errors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, rhs) ||
                stack.pop(errs, lhs) ||
                type.type_obj->idx(errs, heap, dst, lhs, rhs) ||
                stack.push(errs, dst);
        }

        inline bool exec_xstr(Errors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, src, dst;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, src) ||
                type.type_obj->xstr(errs, heap, dst, src) ||
                stack.push(errs, dst);
        }

        inline bool exec_xflt(Errors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, src, dst;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, src) ||
                type.type_obj->xflt(errs, heap, dst, src) ||
                stack.push(errs, dst);
        }

        inline bool exec_xint(Errors& errs, CallStack& stack, ProgramHeap& heap)
        {
            Obj type, src, dst;
            return
                stack.pop(errs, type) ||
                stack.pop(errs, src) ||
                type.type_obj->xint(errs, heap, dst, src) ||
                stack.push(errs, dst);
        }

        inline bool exec_dsp(Errors& errs, CallStack& stack)
        {
            Obj obj;
            if (stack.pop(errs, obj))
                return true;
            std::cout << *obj.str_obj << std::endl;
            return false;
        }

		bool exec(Errors& errs, CallStack& stack, ProgramHeap& heap, CodeSegPtr init_code, DataSegPtr init_data, size_t init_pc)
		{
            // Initializing the interpreter state
            pc = init_pc;
            code = init_code;
            data = init_data;
            complete = false;
            call_stack.clear();

            InstructionType ty;
            while (!complete)
            {
                ty = *(InstructionType*)(code + pc);
                pc += sizeof(InstructionType);
                switch (ty)
                {
                case InstructionType::JMP:
                    if (exec_jmp(errs))
                        goto runtime_error;
                    break;
                case InstructionType::BRT:
                    if (exec_brt(errs, stack))
                        goto runtime_error;
                    break;
                case InstructionType::BRF:
                    if (exec_brf(errs, stack))
                        goto runtime_error;
                    break;
                case InstructionType::NEW:
                    if (exec_new(errs, stack))
                        goto runtime_error;
                    break;
                case InstructionType::AGG:
                    if (exec_agg(errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::ARR:
                    if (exec_arr(errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::ATY:
                    if (exec_aty(errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::POP:
                    if (exec_pop(errs, stack))
                        goto runtime_error;
                    break;
                case InstructionType::DUP:
                    if (exec_dup(errs, stack))
                        goto runtime_error;
                    break;
                case InstructionType::CPY:
                    if (exec_cpy(errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::INST:
                    if (exec_inst(errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::CALL:
                    if (exec_call(errs, stack))
                        goto runtime_error;
                    break;
                case InstructionType::RET:
                    if (exec_ret(errs, stack))
                        goto runtime_error;
                    break;
                case InstructionType::SET:
                    if (exec_set(errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::IADD:
                    if (exec_iadd(errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::ISUB:
                    if (exec_isub(errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::IMUL:
                    if (exec_imul(errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::IDIV:
                    if (exec_idiv(errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::IMOD:
                    if (exec_imod(errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::ADD:
                    if (exec_add(errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::SUB:
                    if (exec_sub(errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::MUL:
                    if (exec_mul(errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::DIV:
                    if (exec_div(errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::MOD:
                    if (exec_mod(errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::EQ:
                    if (exec_eq(errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::NE:
                    if (exec_ne(errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::GT:
                    if (exec_gt(errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::LT:
                    if (exec_lt(errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::GE:
                    if (exec_ge(errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::LE:
                    if (exec_le(errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::IDX:
                    if (exec_idx(errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::XSTR:
                    if (exec_xstr(errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::XFLT:
                    if (exec_xflt(errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::XINT:
                    if (exec_xint(errs, stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::DSP:
                    if (exec_dsp(errs, stack))
                        goto runtime_error;
                    break;
                }
            }
			return false;

        runtime_error:
            // TODO: unwind the call stack with location info
            return true;
		}
	}
}
