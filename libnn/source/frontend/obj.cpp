#include <libnn/frontend/obj.h>

#include <string>

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

        ObjBool::ObjImp(EvalCtx& ctx, const AstDecl* decl, const std::vector<Obj*>& cargs) :
            Obj(ObjType::BOOL)
        {
            ctx.insert({ decl->var_name, this });
            init = false;
            data.val = false;
        }

        bool ObjBool::bval() const
        {
            if (!init)
                throw GenerationError("Uninitialized variable");
            return data.val;
        }

        void ObjBool::assign(const Obj* val)
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

        // input node declaration
        ObjTensor::ObjImp(EvalCtx& ctx, const AstDecl* decl, const std::vector<Obj*>& cargs) :
            Obj(ObjType::TENSOR)
        {
            // reading the tensor dimensions
            for (auto e : cargs)
            {
                if (e->ty != ObjType::INT)
                    throw GenerationError("Unexpected carg type in tensor declaration");
                data.dims.push_back(static_cast<ObjInt*>(e)->getData().val);
            }
            if (data.dims.size() == 0)
                throw GenerationError("A tensor must have at least one carg");

            // adding the variable to the scope
            ctx.insert({ decl->var_name, this });

            // Creating an input edge node
            auto result = varcounts.find(decl->var_name);
            int count = 0;
            if (result == varcounts.end())
                varcounts.insert({ decl->var_name, 1 });
            else
                count = result->second++;
            
            Edge* pEdge = new Edge();
            pEdge->is_static = decl->is_static;
            pEdge->name = decl->var_name + ':' + std::to_string(count);
            pEdge->dsc.rk = data.dims.size();
            pEdge->dsc.dims = data.dims;
            pEdge->input = nullptr;
            pEdge->outputs = {};
            ctx.pgraph->inputs.push_back(pEdge);
            data.pEdge = pEdge;

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
