#include <libnn/frontend/obj.h>

#include <string>
#include <cassert>

namespace nn
{
    namespace impl
    {
        void check_init(const Obj* pobj)
        {
            if (!pobj->init) throw GenerationError("Uninitialized variable");
        }

        void check_init(const std::unique_ptr<Obj>& pobj)
        {
            if (!pobj->init) throw GenerationError("Uninitialized variable");
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

        ObjBool::~ObjImp() {}
        ObjInt::~ObjImp() {}
        ObjFloat::~ObjImp() {}
        ObjStr::~ObjImp() {}
        ObjTuple::~ObjImp() {}
        ObjTensor::~ObjImp() {}

        ObjBool::ObjImp() :
            Obj(ObjType::BOOL)
        {
            init = false;
            data.val = false;
        }

        bool ObjBool::bval() const
        {
            check_init(this);
            return data.val;
        }

        void ObjBool::assign(const std::unique_ptr<Obj>& val)
        {
            check_type(val);
            check_init(val);
            data.val = mty(val)->data.val;
            init = true;
        }

        std::unique_ptr<Obj> ObjBool::andop(const std::unique_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);
            check_type(val);

            std::unique_ptr<ObjBool> nobj = std::make_unique<ObjBool>();
            nobj->data.val = data.val && mty(val)->data.val;
            nobj->init = true;
            return nobj;
        }

        std::unique_ptr<Obj> ObjBool::orop(const std::unique_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);
            check_type(val);

            std::unique_ptr<ObjBool> nobj = std::make_unique<ObjBool>();
            nobj->data.val = data.val || mty(val)->data.val;
            nobj->init = true;
            return nobj;
        }

        std::unique_ptr<Obj> ObjBool::eq(const std::unique_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);
            check_type(val);

            std::unique_ptr<ObjBool> nobj = std::make_unique<ObjBool>();
            nobj->data.val = data.val == mty(val)->data.val;
            nobj->init = true;
            return nobj;
        }

        std::unique_ptr<Obj> ObjBool::ne(const std::unique_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);
            check_type(val);

            std::unique_ptr<ObjBool> nobj = std::make_unique<ObjBool>();
            nobj->data.val = data.val != mty(val)->data.val;
            nobj->init = true;
            return nobj;
        }

        ObjInt::ObjImp() :
            Obj(ObjType::INT)
        {
            init = false;
            data.val = 0;
        }

        void ObjInt::assign(const std::unique_ptr<Obj>& val)
        {
            check_type(val);
            check_init(val);
            data.val = mty(val)->data.val;
            init = true;
        }

        std::unique_ptr<Obj> ObjInt::neg() const
        {
            check_init(this);
            std::unique_ptr<ObjInt> nobj = std::make_unique<ObjInt>();
            nobj->data.val = -data.val;
            nobj->init = true;
            return nobj;
        }

        std::unique_ptr<Obj> ObjInt::add(const std::unique_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);

            if (val->ty == ObjType::INT)
            {
                std::unique_ptr<ObjInt> nobj = std::make_unique<ObjInt>();
                nobj->data.val = data.val + static_cast<ObjInt*>(val.get())->data.val;
                nobj->init = true;
                return nobj;
            }
            else if (val->ty == ObjType::FLOAT)
            {
                std::unique_ptr<ObjFloat> nobj = std::make_unique<ObjFloat>();
                nobj->data.val = data.val + static_cast<ObjFloat*>(val.get())->data.val;
                nobj->init = true;
                return nobj;
            }
            else
                throw GenerationError("Expected int or float, recieved " + objTypeName(val->ty));
        }

        std::unique_ptr<Obj> ObjInt::sub(const std::unique_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);

            if (val->ty == ObjType::INT)
            {
                std::unique_ptr<ObjInt> nobj = std::make_unique<ObjInt>();
                nobj->data.val = data.val - static_cast<ObjInt*>(val.get())->data.val;
                nobj->init = true;
                return nobj;
            }
            else if (val->ty == ObjType::FLOAT)
            {
                std::unique_ptr<ObjFloat> nobj = std::make_unique<ObjFloat>();
                nobj->data.val = data.val - static_cast<ObjFloat*>(val.get())->data.val;
                nobj->init = true;
                return nobj;
            }
            else
                throw GenerationError("Expected int or float, recieved " + objTypeName(val->ty));
        }

        std::unique_ptr<Obj> ObjInt::mul(const std::unique_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);

            if (val->ty == ObjType::INT)
            {
                std::unique_ptr<ObjInt> nobj = std::make_unique<ObjInt>();
                nobj->data.val = data.val * static_cast<ObjInt*>(val.get())->data.val;
                nobj->init = true;
                return nobj;
            }
            else if (val->ty == ObjType::FLOAT)
            {
                std::unique_ptr<ObjFloat> nobj = std::make_unique<ObjFloat>();
                nobj->data.val = data.val * static_cast<ObjFloat*>(val.get())->data.val;
                nobj->init = true;
                return nobj;
            }
            else
                throw GenerationError("Expected int or float, recieved " + objTypeName(val->ty));
        }

        std::unique_ptr<Obj> ObjInt::div(const std::unique_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);

            if (val->ty == ObjType::INT)
            {
                std::unique_ptr<ObjInt> nobj = std::make_unique<ObjInt>();
                nobj->data.val = data.val / static_cast<ObjInt*>(val.get())->data.val;
                nobj->init = true;
                return nobj;
            }
            else if (val->ty == ObjType::FLOAT)
            {
                std::unique_ptr<ObjFloat> nobj = std::make_unique<ObjFloat>();
                nobj->data.val = data.val / static_cast<ObjFloat*>(val.get())->data.val;
                nobj->init = true;
                return nobj;
            }
            else
                throw GenerationError("Expected int or float, recieved " + objTypeName(val->ty));
        }

        std::unique_ptr<Obj> ObjInt::eq(const std::unique_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);

            std::unique_ptr<ObjBool> nobj = std::make_unique<ObjBool>();
            nobj->init = true;
            switch (val->ty)
            {
            case ObjType::INT:
                nobj->data.val = data.val == static_cast<ObjInt*>(val.get())->data.val;
                return nobj;
            case ObjType::FLOAT:
                nobj->data.val = data.val == static_cast<ObjFloat*>(val.get())->data.val;
                return nobj;
            }
            throw GenerationError("Expected int or float, recieved " + objTypeName(val->ty));
        }

        std::unique_ptr<Obj> ObjInt::ne(const std::unique_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);

            std::unique_ptr<ObjBool> nobj = std::make_unique<ObjBool>();
            nobj->init = true;
            switch (val->ty)
            {
            case ObjType::INT:
                nobj->data.val = data.val != static_cast<ObjInt*>(val.get())->data.val;
                return nobj;
            case ObjType::FLOAT:
                nobj->data.val = data.val != static_cast<ObjFloat*>(val.get())->data.val;
                return nobj;
            }
            throw GenerationError("Expected int or float, recieved " + objTypeName(val->ty));
        }

        std::unique_ptr<Obj> ObjInt::ge(const std::unique_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);

            std::unique_ptr<ObjBool> nobj = std::make_unique<ObjBool>();
            nobj->init = true;
            switch (val->ty)
            {
            case ObjType::INT:
                nobj->data.val = data.val >= static_cast<ObjInt*>(val.get())->data.val;
                return nobj;
            case ObjType::FLOAT:
                nobj->data.val = data.val >= static_cast<ObjFloat*>(val.get())->data.val;
                return nobj;
            }
            throw GenerationError("Expected int or float, recieved " + objTypeName(val->ty));
        }

        std::unique_ptr<Obj> ObjInt::le(const std::unique_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);

            std::unique_ptr<ObjBool> nobj = std::make_unique<ObjBool>();
            nobj->init = true;
            switch (val->ty)
            {
            case ObjType::INT:
                nobj->data.val = data.val <= static_cast<ObjInt*>(val.get())->data.val;
                return nobj;
            case ObjType::FLOAT:
                nobj->data.val = data.val <= static_cast<ObjFloat*>(val.get())->data.val;
                return nobj;
            }
            throw GenerationError("Expected int or float, recieved " + objTypeName(val->ty));
        }

        std::unique_ptr<Obj> ObjInt::gt(const std::unique_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);

            std::unique_ptr<ObjBool> nobj = std::make_unique<ObjBool>();
            nobj->init = true;
            switch (val->ty)
            {
            case ObjType::INT:
                nobj->data.val = data.val > static_cast<ObjInt*>(val.get())->data.val;
                return nobj;
            case ObjType::FLOAT:
                nobj->data.val = data.val > static_cast<ObjFloat*>(val.get())->data.val;
                return nobj;
            }
            throw GenerationError("Expected int or float, recieved " + objTypeName(val->ty));
        }

        std::unique_ptr<Obj> ObjInt::lt(const std::unique_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);

            std::unique_ptr<ObjBool> nobj = std::make_unique<ObjBool>();
            nobj->init = true;
            switch (val->ty)
            {
            case ObjType::INT:
                nobj->data.val = data.val < static_cast<ObjInt*>(val.get())->data.val;
                return nobj;
            case ObjType::FLOAT:
                nobj->data.val = data.val < static_cast<ObjFloat*>(val.get())->data.val;
                return nobj;
            }
            throw GenerationError("Expected int or float, recieved " + objTypeName(val->ty));
        }

        ObjFloat::ObjImp() :
            Obj(ObjType::FLOAT)
        {
            init = false;
            data.val = 1.0;
        }

        ObjTensor::ObjImp() :
            Obj(ObjType::TENSOR)
        {
            init = false;
            data.dims = {};
            data.pEdge = nullptr;
        }

        void ObjTensor::assign(const std::unique_ptr<Obj>& val)
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

        void ObjTensor::cargs(const std::vector<std::unique_ptr<Obj>>& cargs)
        {
            if (data.carg_init)
                throw GenerationError("Cannot apply constant args to a pre-defined tensor");

            // reading the tensor dimensions
            for (auto& e : cargs)
            {
                if (e->ty != ObjType::INT)
                    throw GenerationError("Unexpected carg type in tensor declaration");
                data.dims.push_back(static_cast<ObjInt*>(e.get())->data.val);
            }
            if (data.dims.size() == 0)
                throw GenerationError("A tensor must have at least one carg");

            data.carg_init = true;
        }
    }
}
