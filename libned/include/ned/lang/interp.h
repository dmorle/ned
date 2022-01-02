#ifndef NED_INTERP_H
#define NED_INTERP_H

#include <ned/lang/errors.h>
#include <ned/lang/obj.h>

namespace nn
{
    namespace lang
    {
        class CodeSegment
        {

        };

        class CallStack
        {
            std::array<Obj, 1e6> stack;
            size_t sp;

        public:
            bool pop(RuntimeErrors& errs, Obj& obj);
            bool get(RuntimeErrors& errs, size_t i, Obj& obj);
            bool push(RuntimeErrors& errs, Obj val);
        };
    }
}

#endif
