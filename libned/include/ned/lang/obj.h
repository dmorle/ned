#ifndef NED_OBJ_H
#define NED_OBJ_H

#include <string>
#include <sstream>
#include <unordered_map>

#include <ned/core/graph.h>
#include <ned/lang/ast.h>
#include <ned/lang/interp.h>

namespace nn
{
    namespace lang
    {
        enum class ObjType
        {
            TYPE,     // Type object
            BOOL,     // Boolean
            FWIDTH,   // Float widths for tensors, ie f16, f32, f64
            INT,      // Integer
            FLOAT,    // Floating point
            STR,      // String
            AGG,      // Aggregate types: arrays, tuples, structs
            TENSOR    // Tensor - Graph edge
        };

        struct Type
        {
            virtual bool inst (RuntimeErrors& errs, CallStack& stack) = 0;
            virtual bool set  (RuntimeErrors& errs, CallStack& stack) = 0;
            virtual bool add  (RuntimeErrors& errs, CallStack& stack) = 0;
            virtual bool sub  (RuntimeErrors& errs, CallStack& stack) = 0;
            virtual bool mul  (RuntimeErrors& errs, CallStack& stack) = 0;
            virtual bool div  (RuntimeErrors& errs, CallStack& stack) = 0;
            virtual bool mod  (RuntimeErrors& errs, CallStack& stack) = 0;
            virtual bool eq   (RuntimeErrors& errs, CallStack& stack) = 0;
            virtual bool ne   (RuntimeErrors& errs, CallStack& stack) = 0;
            virtual bool ge   (RuntimeErrors& errs, CallStack& stack) = 0;
            virtual bool le   (RuntimeErrors& errs, CallStack& stack) = 0;
            virtual bool gt   (RuntimeErrors& errs, CallStack& stack) = 0;
            virtual bool lt   (RuntimeErrors& errs, CallStack& stack) = 0;
            virtual bool land (RuntimeErrors& errs, CallStack& stack) = 0;
            virtual bool lor  (RuntimeErrors& errs, CallStack& stack) = 0;
            virtual bool idx  (RuntimeErrors& errs, CallStack& stack) = 0;
            virtual bool xstr (RuntimeErrors& errs, CallStack& stack) = 0;
            virtual bool xflt (RuntimeErrors& errs, CallStack& stack) = 0;
            virtual bool xint (RuntimeErrors& errs, CallStack& stack) = 0;
            virtual bool cblk (RuntimeErrors& errs, CallStack& stack) = 0;
            virtual bool iblk (RuntimeErrors& errs, CallStack& stack) = 0;
            virtual bool binp (RuntimeErrors& errs, CallStack& stack) = 0;
            virtual bool bout (RuntimeErrors& errs, CallStack& stack) = 0;
            virtual bool ext  (RuntimeErrors& errs, CallStack& stack) = 0;
            virtual bool exp  (RuntimeErrors& errs, CallStack& stack) = 0;
        };

        union Obj
        {
            Type             *type_obj;
            bool             *bool_obj;
            core::tensor_dty *fwidth_obj;
            int64_t          *int_obj;
            double           *float_obj;
            std::string      *str_obj;
            std::vector<Obj> *agg_obj;
            core::Edge       *ten_obj;
            void*             ptr;
        };

        struct BoolType
        {

        };

        struct FWidthType
        {

        };

        struct IntType
        {

        };

        struct FloatType
        {

        };

        struct StringType
        {

        };

        struct AggType
        {

        };
    }
}

#endif
