#include <libnn/frontend/obj.h>

#include <string>
#include <cassert>

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

        ObjBool::~ObjImp() {}
        ObjInt::~ObjImp() {}
        ObjFloat::~ObjImp() {}
        ObjStr::~ObjImp() {}
        ObjTuple::~ObjImp() {}
        ObjTensor::~ObjImp() {}

        ObjBool::ObjImp(const std::vector<std::unique_ptr<Obj>>& cargs) :
            Obj(ObjType::BOOL)
        {
            if (cargs.size() != 0)
                throw GenerationError("bool type does not accept any cargs");
            init = false;
            data.val = false;
        }

        bool ObjBool::bval() const
        {
            if (!init)
                throw GenerationError("Uninitialized variable");
            return data.val;
        }

        void ObjBool::assign(const std::unique_ptr<Obj> val)
        {
            if (val->ty != ObjType::BOOL)
                throw GenerationError("Unable to assign type " + objTypeName(val->ty) + " to type " + type_name);

            data.val = mty(val)->data.val;
            init = true;
        }

        bool ObjInt::bval() const
        {
            if (!init)
                throw GenerationError("Uninitialized variable"); 
            throw GenerationError("'int' variable does not have a truth value");
        }

        bool ObjFloat::bval() const
        {
            if (!init)
                throw GenerationError("Uninitialized variable"); 
            throw GenerationError("'float' variable does not have a truth value");
        }

        bool ObjStr::bval() const
        {
            if (!init)
                throw GenerationError("Uninitialized variable"); 
            throw GenerationError("'str' variable does not have a truth value");
        }

        bool ObjTuple::bval() const
        {
            if (!init)
                throw GenerationError("Uninitialized variable"); 
            throw GenerationError("'tuple' variable does not have a truth value");
        }

        static std::unordered_map<std::string, int> varcounts = {};

        ObjTensor::ObjImp(const std::vector<std::unique_ptr<Obj>>& cargs) :
            Obj(ObjType::TENSOR)
        {
            // reading the tensor dimensions
            for (auto& e : cargs)
            {
                if (e->ty != ObjType::INT)
                    throw GenerationError("Unexpected carg type in tensor declaration");
                data.dims.push_back(static_cast<ObjInt*>(e.get())->getData().val);
            }
            if (data.dims.size() == 0)
                throw GenerationError("A tensor must have at least one carg");

            // creating the new edge
            data.pEdge = new Edge();
            data.pEdge->is_static = false;
            data.pEdge->dsc.rk = data.dims.size();
            data.pEdge->dsc.dims = data.dims;
            data.pEdge->input = nullptr;
            data.pEdge->inpid = -1;
            data.pEdge->outputs = {};
            init = true;
        }

        ObjTensor::ObjImp(const std::vector<std::unique_ptr<Obj>>& cargs, const ObjData<ObjType::TENSOR>& val) :
            Obj(ObjType::TENSOR)
        {
            // reading the tensor dimensions
            for (auto& e : cargs)
            {
                if (e->ty != ObjType::INT)
                    throw GenerationError("Unexpected carg type in tensor declaration");
                data.dims.push_back(static_cast<ObjInt*>(e.get())->getData().val);
            }
            if (data.dims.size() == 0)
                throw GenerationError("A tensor must have at least one carg");

            assert(val.dims.size() != 0);  // uninitialized

            data.pEdge = val.pEdge;
            init = true;
        }

        bool ObjTensor::bval() const
        {
            if (!init)
                throw GenerationError("Uninitialized variable");
            throw GenerationError("'tensor' variable does not have a truth value");
        }
    }
}
