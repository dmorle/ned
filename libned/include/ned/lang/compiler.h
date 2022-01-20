#ifndef NED_TYPING_H
#define NED_TYPING_H

#include <ned/lang/ast.h>
#include <ned/lang/bytecode.h>

namespace nn
{
    namespace lang
    {
        enum class TypeEnum
        {
            TYPE,       // Type object
            BOOL,       // Boolean
            FWIDTH,     // Float widths for tensors, ie f16, f32, f64
            INT,        // Integer
            FLOAT,      // Floating point
            STR,        // String
            ARRAY,      // Array - Fixed length
            TUPLE,      // Tuple - mainly for cargs
            NAMESPACE,  // Reference to a namespace
            STRUCT,     // Generation time struct
            FN,         // Generation time function
            DEF,        // Compound block reference
            INTR,       // Intrinsic block reference
            EDGE,       // Either forward or backward edge
            TENSOR,     // Forward and backward edge
            GENERIC,    // Compile time generic type
            PARAMPACK   // A packed set of parameters
        };

        bool codegen(ByteCodeModule& bc, const AstModule& ast, const std::vector<std::string>& imp_dirs);
    }
}

#endif
