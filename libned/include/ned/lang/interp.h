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
        using BlockOffsets = std::map<std::string, size_t>;

        class CallStack
        {
            std::array<Obj, (size_t)1e6> stack;
            size_t sp;

        public:
            bool pop(Errors& errs, Obj& obj);
            bool del(Errors& errs, size_t i);
            bool get(Errors& errs, size_t i, Obj& obj);
            bool push(Errors& errs, Obj obj);
        };

        bool exec(Errors& errs, CallStack& stack, ProgramHeap& heap, CodeSegPtr code, DataSegPtr data, size_t pc);
    }
}

#endif
