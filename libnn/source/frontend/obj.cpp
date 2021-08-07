#include <libnn/frontend/obj.h>

#include <string>
#include <cassert>

namespace nn
{
    namespace impl
    {
        void check_init(const Obj* pobj)
        {
            if (!pobj->init)
                throw GenerationError("Uninitialized variable");
        }

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

        template<> ObjBool::~ObjImp() {}
        template<> ObjInt::~ObjImp() {}
        template<> ObjFloat::~ObjImp() {}
        template<> ObjStr::~ObjImp() {}
        template<> ObjTuple::~ObjImp() {}
        template<> ObjTensor::~ObjImp() {}

        template<>
        ObjBool::ObjImp() :
            Obj(ObjType::BOOL)
        {
            init = false;
            data.val = false;
        }

        template<>
        bool ObjBool::bval() const
        {
            check_init(this);
            return data.val;
        }

        template<>
        void ObjBool::assign(const Obj* val)
        {
            check_type(val);
            check_init(val);
            data.val = mty(val)->data.val;
            init = true;
        }

        template<>
        Obj* ObjBool::andop(const Obj* val) const
        {
            check_init(this);
            check_init(val);
            check_type(val);

            ObjBool* nobj = new ObjBool();
            nobj->data.val = data.val && mty(val)->data.val;
            nobj->init = true;
            return nobj;
        }

        template<>
        Obj* ObjBool::orop(const Obj* val) const
        {
            check_init(this);
            check_init(val);
            check_type(val);

            ObjBool* nobj = new ObjBool();
            nobj->data.val = data.val || mty(val)->data.val;
            nobj->init = true;
            return nobj;
        }

        template<>
        Obj* ObjBool::eq(const Obj* val) const
        {
            check_init(this);
            check_init(val);
            check_type(val);

            ObjBool* nobj = new ObjBool();
            nobj->data.val = data.val == mty(val)->data.val;
            nobj->init = true;
            return nobj;
        }

        template<>
        Obj* ObjBool::ne(const Obj* val) const
        {
            check_init(this);
            check_init(val);
            check_type(val);

            ObjBool* nobj = new ObjBool();
            nobj->data.val = data.val != mty(val)->data.val;
            nobj->init = true;
            return nobj;
        }

        template<>
        ObjInt::ObjImp() :
            Obj(ObjType::INT)
        {
            init = false;
            data.val = 0;
        }

        template<>
        void ObjInt::assign(const Obj* val)
        {
            check_type(val);
            check_init(val);
            data.val = mty(val)->data.val;
            init = true;
        }

        template<>
        Obj* ObjInt::neg() const
        {
            check_init(this);
            ObjInt* nobj = new ObjInt();
            nobj->data.val = -data.val;
            nobj->init = true;
            return nobj;
        }

        template<>
        Obj* ObjInt::add(const Obj* val) const
        {
            check_init(this);
            check_init(val);

            if (val->ty == ObjType::INT)
            {
                ObjInt* nobj = new ObjInt();
                nobj->data.val = data.val + static_cast<const ObjInt*>(val)->data.val;
                nobj->init = true;
                return nobj;
            }
            else if (val->ty == ObjType::FLOAT)
            {
                ObjFloat* nobj = new ObjFloat();
                nobj->data.val = data.val + static_cast<const ObjFloat*>(val)->data.val;
                nobj->init = true;
                return nobj;
            }
            else
                throw GenerationError("Expected int or float, recieved " + objTypeName(val->ty));
        }

        template<>
        Obj* ObjInt::sub(const Obj* val) const
        {
            check_init(this);
            check_init(val);

            if (val->ty == ObjType::INT)
            {
                ObjInt* nobj = new ObjInt();
                nobj->data.val = data.val - static_cast<const ObjInt*>(val)->data.val;
                nobj->init = true;
                return nobj;
            }
            else if (val->ty == ObjType::FLOAT)
            {
                ObjFloat* nobj = new ObjFloat();
                nobj->data.val = data.val - static_cast<const ObjFloat*>(val)->data.val;
                nobj->init = true;
                return nobj;
            }
            else
                throw GenerationError("Expected int or float, recieved " + objTypeName(val->ty));
        }

        template<>
        Obj* ObjInt::mul(const Obj* val) const
        {
            check_init(this);
            check_init(val);

            if (val->ty == ObjType::INT)
            {
                ObjInt* nobj = new ObjInt();
                nobj->data.val = data.val * static_cast<const ObjInt*>(val)->data.val;
                nobj->init = true;
                return nobj;
            }
            else if (val->ty == ObjType::FLOAT)
            {
                ObjFloat* nobj = new ObjFloat();
                nobj->data.val = data.val * static_cast<const ObjFloat*>(val)->data.val;
                nobj->init = true;
                return nobj;
            }
            else
                throw GenerationError("Expected int or float, recieved " + objTypeName(val->ty));
        }

        template<>
        Obj* ObjInt::div(const Obj* val) const
        {
            check_init(this);
            check_init(val);

            if (val->ty == ObjType::INT)
            {
                ObjInt* nobj = new ObjInt();
                nobj->data.val = data.val / static_cast<const ObjInt*>(val)->data.val;
                nobj->init = true;
                return nobj;
            }
            else if (val->ty == ObjType::FLOAT)
            {
                ObjFloat* nobj = new ObjFloat();
                nobj->data.val = data.val / static_cast<const ObjFloat*>(val)->data.val;
                nobj->init = true;
                return nobj;
            }
            else
                throw GenerationError("Expected int or float, recieved " + objTypeName(val->ty));
        }

        template<>
        Obj* ObjInt::eq(const Obj* val) const
        {
            check_init(this);
            check_init(val);

            ObjBool* nobj = new ObjBool();
            nobj->init = true;
            switch (val->ty)
            {
            case ObjType::INT:
                nobj->data.val = data.val == static_cast<const ObjInt*>(val)->data.val;
                return nobj;
            case ObjType::FLOAT:
                nobj->data.val = data.val == static_cast<const ObjFloat*>(val)->data.val;
                return nobj;
            }
            throw GenerationError("Expected int or float, recieved " + objTypeName(val->ty));
        }

        template<>
        Obj* ObjInt::ne(const Obj* val) const
        {
            check_init(this);
            check_init(val);

            ObjBool* nobj = new ObjBool();
            nobj->init = true;
            switch (val->ty)
            {
            case ObjType::INT:
                nobj->data.val = data.val != static_cast<const ObjInt*>(val)->data.val;
                return nobj;
            case ObjType::FLOAT:
                nobj->data.val = data.val != static_cast<const ObjFloat*>(val)->data.val;
                return nobj;
            }
            throw GenerationError("Expected int or float, recieved " + objTypeName(val->ty));
        }

        template<>
        Obj* ObjInt::ge(const Obj* val) const
        {
            check_init(this);
            check_init(val);

            ObjBool* nobj = new ObjBool();
            nobj->init = true;
            switch (val->ty)
            {
            case ObjType::INT:
                nobj->data.val = data.val >= static_cast<const ObjInt*>(val)->data.val;
                return nobj;
            case ObjType::FLOAT:
                nobj->data.val = data.val >= static_cast<const ObjFloat*>(val)->data.val;
                return nobj;
            }
            throw GenerationError("Expected int or float, recieved " + objTypeName(val->ty));
        }

        template<>
        Obj* ObjInt::le(const Obj* val) const
        {
            check_init(this);
            check_init(val);

            ObjBool* nobj = new ObjBool();
            nobj->init = true;
            switch (val->ty)
            {
            case ObjType::INT:
                nobj->data.val = data.val <= static_cast<const ObjInt*>(val)->data.val;
                return nobj;
            case ObjType::FLOAT:
                nobj->data.val = data.val <= static_cast<const ObjFloat*>(val)->data.val;
                return nobj;
            }
            throw GenerationError("Expected int or float, recieved " + objTypeName(val->ty));
        }

        template<>
        Obj* ObjInt::gt(const Obj* val) const
        {
            check_init(this);
            check_init(val);

            ObjBool* nobj = new ObjBool();
            nobj->init = true;
            switch (val->ty)
            {
            case ObjType::INT:
                nobj->data.val = data.val > static_cast<const ObjInt*>(val)->data.val;
                return nobj;
            case ObjType::FLOAT:
                nobj->data.val = data.val > static_cast<const ObjFloat*>(val)->data.val;
                return nobj;
            }
            throw GenerationError("Expected int or float, recieved " + objTypeName(val->ty));
        }

        template<>
        Obj* ObjInt::lt(const Obj* val) const
        {
            check_init(this);
            check_init(val);

            ObjBool* nobj = new ObjBool();
            nobj->init = true;
            switch (val->ty)
            {
            case ObjType::INT:
                nobj->data.val = data.val < static_cast<const ObjInt*>(val)->data.val;
                return nobj;
            case ObjType::FLOAT:
                nobj->data.val = data.val < static_cast<const ObjFloat*>(val)->data.val;
                return nobj;
            }
            throw GenerationError("Expected int or float, recieved " + objTypeName(val->ty));
        }

        template<>
        ObjFloat::ObjImp() :
            Obj(ObjType::FLOAT)
        {
            init = false;
            data.val = 1.0;
        }

        template<>
        ObjTensor::ObjImp() :
            Obj(ObjType::TENSOR)
        {
            init = false;
            data.dims = {};
            data.pEdge = nullptr;
        }

        void ObjTensor::assign(const Obj* val)
        {
            check_type(val);
            check_init(val);

            if (data.carg_init)
            {
                // check to ensure the new value has the same rank and dimensions
                if (data.dims.size() != mty(val)->data.dims.size())
                    throw GenerationError(
                        "Unable to assign a tensor with rank " + std::to_string(mty(val)->data.dims.size()) +
                        " to a tensor with rank "              + std::to_string(          data.dims.size()));
                for (size_t i = 0; i < data.dims.size(); i++)
                {
                    if (data.dims[i] != mty(val)->data.dims[i])
                        throw GenerationError(
                            "Unable to assign a tensor with dimension " + std::to_string(mty(val)->data.dims[i]) + " on index " + std::to_string(i) +
                            " to a tensor with dimension " + std::to_string(data.dims.size()) + " on index " + std::to_string(i));
                }
            }
            else
            {
                // initializing data.dims
                assert(data.dims.size() == 0);
                for (auto e : mty(val)->data.dims)
                    data.dims.push_back(e);
                data.carg_init = true;
            }

            // doing the actual assignment
            data.pEdge = mty(val)->data.pEdge;
            init = true;
        }

        template<>
        Obj* ObjTensor::cargs(const std::vector<Obj*>& cargs)
        {
            if (data.carg_init)
                throw GenerationError("Cannot apply constant args to a pre-defined tensor");

            // reading the tensor dimensions
            for (auto& e : cargs)
            {
                if (e->ty != ObjType::INT)
                    throw GenerationError("Unexpected carg type in tensor declaration");
                data.dims.push_back(static_cast<ObjInt*>(e)->data.val);
            }
            if (data.dims.size() == 0)
                throw GenerationError("A tensor must have at least one carg");

            data.carg_init = true;
        }
    }
}
