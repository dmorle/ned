#include <libnn/frontend/obj.h>

namespace nn
{
    namespace impl
    {
        constexpr std::string objTypeName(ObjType ty)
        {
            switch(ty)
            {
            case ObjType::INVALID:
                return "invalid";
            case ObjType::BOOL:
                return "bool";
            case ObjType::INT:
                return "int";
            case ObjType::FLOAT:
                return "float";
            case ObjType::STR:
                return "string";
            case ObjType::TUPLE:
                return "tuple";
            case ObjType::TENSOR:
                return "tensor";
            case ObjType::INTR:
                return "intrinsic";
            case ObjType::DEF:
                return "def";
            case ObjType::MODULE:
                return "module";
            case ObjType::PACKAGE:
                return "package";
            }
        }

        Obj::Obj(ObjType ty) :
            ty(ty),
            init(false)
        {}

        template<> ObjImp<ObjType::BOOL>::~ObjImp() {}
        template<> ObjImp<ObjType::INT>::~ObjImp() {}
        template<> ObjImp<ObjType::FLOAT>::~ObjImp() {}
        template<> ObjImp<ObjType::STR>::~ObjImp() {}
        template<> ObjImp<ObjType::TUPLE>::~ObjImp() {}
        template<> ObjImp<ObjType::TENSOR>::~ObjImp() {}

        template<> bool ObjImp<ObjType::BOOL>::bval() const
        {
            if (!init)
                throw GenerationError("Uninitialized variable");
            return data.val;
        }

        template<> void ObjImp<ObjType::BOOL>::assign(const Obj* val)
        {
            if (val->ty != ObjType::BOOL)
                throw GenerationError("Unable to assign type " + objTypeName(val->ty) + " to type " + type_name);

            data.val = mty(val)->data.val;
            init = true;
        }

        template<> bool ObjImp<ObjType::INT>::bval() const
        {
            if (!init)
                throw GenerationError("Uninitialized variable"); 
            throw GenerationError("'int' variable does not have a truth value");
        }

        template<> bool ObjImp<ObjType::FLOAT>::bval() const
        {
            if (!init)
                throw GenerationError("Uninitialized variable"); 
            throw GenerationError("'float' variable does not have a truth value");
        }

        template<> bool ObjImp<ObjType::STR>::bval() const
        {
            if (!init)
                throw GenerationError("Uninitialized variable"); 
            throw GenerationError("'str' variable does not have a truth value");
        }

        template<> bool ObjImp<ObjType::TUPLE>::bval() const
        {
            if (!init)
                throw GenerationError("Uninitialized variable"); 
            throw GenerationError("'tuple' variable does not have a truth value");
        }

        template<> bool ObjImp<ObjType::TENSOR>::bval() const
        {
            if (!init)
                throw GenerationError("Uninitialized variable");
            throw GenerationError("'tensor' variable does not have a truth value");
        }
    }
}
