#ifndef NED_TYPING_H
#define NED_TYPING_H

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
            ARRAY,    // Array - Fixed length
            TUPLE,    // Tuple - mainly for cargs
            STRUCT,   // Generation time struct
            FN,       // Generation time function
            DEF,      // Compound block reference
            INTR,     // Intrinsic block reference
            TENSOR    // Tensor - Graph edge
        };

        template<typename T>
        concept ObjectRef = requires(T a)
        {
            { a.get_type() } -> ObjType;
        };
    }
}

#endif
