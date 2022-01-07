#ifndef NED_INTERP_H
#define NED_INTERP_H

#include <ned/lang/errors.h>
#include <ned/lang/obj.h>

#include <array>

namespace nn
{
    namespace lang
    {
        using CodeSegPtr = uint8_t*;
        using DataSegPtr = Obj*;

        class CallStack
        {
            std::array<Obj, (size_t)1e6> stack;
            size_t sp;

        public:
            bool pop(Errors& errs, Obj& obj);
            bool get(Errors& errs, size_t i, Obj& obj);
            bool push(Errors& errs, Obj val);
        };

        bool exec(Errors& errs, CallStack& stack, ProgramHeap& heap, CodeSegPtr* code, DataSegPtr* data, size_t pc);
    }
}

#endif
