#include <ned/lang/obj.h>

#include <string>
#include <cassert>

namespace nn
{
    namespace lang
    {
        void check_init(const Obj* pobj)
        {
            if (!pobj->init)
                throw GenerationError("Uninitialized variable");
        }

        void check_init(const std::shared_ptr<Obj>& pobj)
        {
            if (!pobj->init)
                throw GenerationError("Uninitialized variable");
        }

        void check_type(ObjType expected, const std::shared_ptr<Obj>& pobj)
        {
            if (pobj->ty != expected)
                throw GenerationError("Expected " + obj_type_name(expected) + ", recieved " + obj_type_name(pobj->ty));
        }

        void check_type(ObjType expected, const Obj* pobj)
        {
            if (pobj->ty != expected)
                throw GenerationError("Expected " + obj_type_name(expected) + ", recieved " + obj_type_name(pobj->ty));
        }

        constexpr std::string obj_type_name(ObjType ty)
        {
            switch (ty)
            {
            case ObjType::TYPE:
                return "type";
            case ObjType::INVALID:
                return "invalid";
            case ObjType::VAR:
                return "var";
            case ObjType::BOOL:
                return "bool";
            case ObjType::INT:
                return "int";
            case ObjType::FLOAT:
                return "float";
            case ObjType::STR:
                return "string";
            case ObjType::ARRAY:
                return "array";
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

            throw GenerationError("Unknown type");
        }
        
        Obj::Obj(ObjType ty) :
            ty(ty),
            init(false)
        {}

        template<>
        ObjDType::ObjImp() :
            Obj(ObjType::TYPE)
        {
            data.ety = ObjType::INVALID;
            data.has_cargs = false;
            data.cargs = {};
        }

        template<>
        ObjDType::~ObjImp() {}

        template<>
        std::shared_ptr<Obj> ObjDType::copy() const
        {
            check_init(this);

            if (data.has_cargs)
                return create_obj_dtype(data.ety, data.cargs);
            return create_obj_dtype(data.ety);
        }

        template<>
        std::shared_ptr<Obj> ObjDType::inst() const
        {
            check_init(this);

            switch (data.ety)
            {
            case ObjType::TYPE:
                assert(!data.has_cargs);
                return create_obj_dtype();
            case ObjType::FWIDTH:
                assert(!data.has_cargs);
                return create_obj_fwidth();
            case ObjType::BOOL:
                assert(!data.has_cargs);
                return create_obj_bool();
            case ObjType::INT:
                assert(!data.has_cargs);
                return create_obj_int();
            case ObjType::FLOAT:
                assert(!data.has_cargs);
                return create_obj_float();
            case ObjType::STR:
                assert(!data.has_cargs);
                return create_obj_str();
            case ObjType::ARRAY:
                if (data.has_cargs)
                {
                    if (data.cargs.size() != 2)
                        throw GenerationError("Invalid number of carg parameters for array type");
                    check_type(ObjType::TYPE, data.cargs[0]);
                    check_type(ObjType::INT, data.cargs[1]);
                    return create_obj_array(std::static_pointer_cast<ObjDType>(data.cargs[0]), static_cast<const ObjInt*>(data.cargs[1].get())->data.val);
                }
                else
                    return create_obj_array();
            case ObjType::TUPLE:
                if (data.has_cargs)
                {
                    std::vector<std::shared_ptr<ObjDType>> dtypes;
                    for (auto e : data.cargs)
                    {
                        check_type(ObjType::TYPE, e);
                        dtypes.push_back(std::static_pointer_cast<ObjDType>(e));
                    }
                    return create_obj_tuple(dtypes);
                }
                else
                    return create_obj_tuple();
            case ObjType::TENSOR:
                if (data.has_cargs)
                {
                    if (data.cargs.size() == 0)
                        throw GenerationError("Invalid number of carg parameters for tensor type");
                    check_type(ObjType::FWIDTH, data.cargs[0]);
                    std::vector<uint32_t> dims;
                    for (size_t i = 1; i < data.cargs.size(); i++)
                    {
                        check_init(data.cargs[i]);
                        check_type(ObjType::INT, data.cargs[i]);
                        dims.push_back(std::static_pointer_cast<ObjInt>(data.cargs[i])->data.val);
                    }
                    return create_obj_tensor(std::static_pointer_cast<ObjFWidth>(data.cargs[0])->data.dty, dims);
                }
                else
                    return create_obj_tensor();
            }
            throw GenerationError("Invalid type for instantiation");
        }

        template<>
        std::shared_ptr<Obj> ObjDType::cargs(const std::vector<std::shared_ptr<Obj>>& args)
        {
            check_init(this);
            if (data.has_cargs)
                throw GenerationError("Const args have already been set");
            return create_obj_dtype(data.ety, args);
        }

        template<>
        std::shared_ptr<Obj> ObjDType::eq(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);
            check_mtype(val);

            // shortcut
            if (this == val.get())
                return create_obj_bool(true);

            if (data.ety != mty(val)->data.ety)
                return create_obj_bool(false);
            if (data.has_cargs)
            {
                if (data.cargs.size() != mty(val)->data.cargs.size())
                    return create_obj_bool(false);
                for (int i = 0; i < data.cargs.size(); i++)
                    if (data.cargs[i]->ne(mty(val)->data.cargs[i])->bval())
                        return create_obj_bool(false);
            }
            return create_obj_bool(true);
        }

        template<>
        std::shared_ptr<Obj> ObjDType::ne(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);
            check_mtype(val);

            // shortcut
            if (this == val.get())
                return create_obj_bool(false);

            if (data.ety != mty(val)->data.ety)
                return create_obj_bool(true);
            if (data.has_cargs)
            {
                if (data.cargs.size() != mty(val)->data.cargs.size())
                    return create_obj_bool(true);
                for (int i = 0; i < data.cargs.size(); i++)
                    if (data.cargs[i]->ne(mty(val)->data.cargs[i])->bval())
                        return create_obj_bool(true);
            }
            return create_obj_bool(false);
        }

        template<>
        ObjInvalid::ObjImp() :
            Obj(ObjType::INVALID)
        {}

        template<>
        ObjInvalid::~ObjImp() {}

        template<>
        ObjVar::ObjImp() :
            Obj(ObjType::VAR)
        {
            data.self = nullptr;
        }

        template<>
        ObjVar::~ObjImp() {}

        template<>
        bool ObjVar::bval() const
        {
            if (!data.self)
                return ObjImp::bval();
            return data.self->bval();
        }

        // TODO: implement the rest of the ObjVar methods

        template<>
        ObjFWidth::ObjImp() :
            Obj(ObjType::FWIDTH)
        {
            data.dty = core::tensor_dty::F32;
        }

        template<>
        void ObjFWidth::assign(const std::shared_ptr<Obj>& val)
        {
            check_mtype(val);
            check_init(val);
            data.dty = mty(val)->data.dty;
            init = true;
        }

        template<>
        std::shared_ptr<Obj> ObjFWidth::copy() const
        {
            check_init(this);
            return create_obj_fwidth(data.dty);
        }

        template<>
        ObjBool::ObjImp() :
            Obj(ObjType::BOOL)
        {
            data.val = false;
        }

        template<>
        ObjBool::~ObjImp() {}

        template<>
        bool ObjBool::bval() const
        {
            check_init(this);
            return data.val;
        }

        template<>
        void ObjBool::assign(const std::shared_ptr<Obj>& val)
        {
            check_mtype(val);
            check_init(val);
            data.val = mty(val)->data.val;
            init = true;
        }

        template<>
        std::shared_ptr<Obj> ObjBool::copy() const
        {
            check_init(this);
            return create_obj_bool(data.val);
        }

        template<>
        std::shared_ptr<Obj> ObjBool::andop(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);
            check_mtype(val);

            auto pobj = create_obj_bool();
            pobj->data.val = data.val && mty(val)->data.val;
            pobj->init = true;
            return pobj;
        }

        template<>
        std::shared_ptr<Obj> ObjBool::orop(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);
            check_mtype(val);

            auto pobj = create_obj_bool();
            pobj->data.val = data.val || mty(val)->data.val;
            pobj->init = true;
            return pobj;
        }

        template<>
        std::shared_ptr<Obj> ObjBool::eq(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);
            check_mtype(val);

            auto pobj = create_obj_bool();
            pobj->data.val = data.val == mty(val)->data.val;
            pobj->init = true;
            return pobj;
        }

        template<>
        std::shared_ptr<Obj> ObjBool::ne(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);
            check_mtype(val);

            auto pobj = create_obj_bool();
            pobj->data.val = data.val != mty(val)->data.val;
            pobj->init = true;
            return pobj;
        }

        template<>
        ObjInt::ObjImp() :
            Obj(ObjType::INT)
        {
            data.val = 0;
        }

        template<>
        ObjInt::~ObjImp() {}

        template<>
        void ObjInt::assign(const std::shared_ptr<Obj>& val)
        {
            check_mtype(val);
            check_init(val);
            data.val = mty(val)->data.val;
            init = true;
        }

        template<>
        std::shared_ptr<Obj> ObjInt::copy() const
        {
            check_init(this);
            return create_obj_int(data.val);
        }

        template<>
        std::shared_ptr<Obj> ObjInt::neg() const
        {
            check_init(this);
            auto pobj = create_obj_int();
            pobj->data.val = -data.val;
            pobj->init = true;
            return pobj;
        }

        template<>
        std::shared_ptr<Obj> ObjInt::add(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);

            if (val->ty == ObjType::INT)
            {
                auto pobj = create_obj_int();
                pobj->data.val = data.val + static_cast<const ObjInt*>(val.get())->data.val;
                pobj->init = true;
                return pobj;
            }
            else if (val->ty == ObjType::FLOAT)
            {
                auto pobj = create_obj_float();
                pobj->data.val = data.val + static_cast<const ObjFloat*>(val.get())->data.val;
                pobj->init = true;
                return pobj;
            }
            else
                throw GenerationError("Expected int or float, recieved " + obj_type_name(val->ty));
        }

        template<>
        std::shared_ptr<Obj> ObjInt::sub(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);

            if (val->ty == ObjType::INT)
            {
                auto pobj = create_obj_int();
                pobj->data.val = data.val - static_cast<const ObjInt*>(val.get())->data.val;
                pobj->init = true;
                return pobj;
            }
            else if (val->ty == ObjType::FLOAT)
            {
                auto pobj = create_obj_float();
                pobj->data.val = data.val - static_cast<const ObjFloat*>(val.get())->data.val;
                pobj->init = true;
                return pobj;
            }
            else
                throw GenerationError("Expected int or float, recieved " + obj_type_name(val->ty));
        }

        template<>
        std::shared_ptr<Obj> ObjInt::mul(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);

            if (val->ty == ObjType::INT)
            {
                auto pobj = create_obj_int();
                pobj->data.val = data.val * static_cast<const ObjInt*>(val.get())->data.val;
                pobj->init = true;
                return pobj;
            }
            else if (val->ty == ObjType::FLOAT)
            {
                auto pobj = create_obj_float();
                pobj->data.val = data.val * static_cast<const ObjFloat*>(val.get())->data.val;
                pobj->init = true;
                return pobj;
            }
            else
                throw GenerationError("Expected int or float, recieved " + obj_type_name(val->ty));
        }

        template<>
        std::shared_ptr<Obj> ObjInt::div(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);

            if (val->ty == ObjType::INT)
            {
                auto pobj = create_obj_int();
                pobj->data.val = data.val / static_cast<const ObjInt*>(val.get())->data.val;
                pobj->init = true;
                return pobj;
            }
            else if (val->ty == ObjType::FLOAT)
            {
                auto pobj = create_obj_float();
                pobj->data.val = data.val / static_cast<const ObjFloat*>(val.get())->data.val;
                pobj->init = true;
                return pobj;
            }
            else
                throw GenerationError("Expected int or float, recieved " + obj_type_name(val->ty));
        }

        template<>
        std::shared_ptr<Obj> ObjInt::eq(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);

            auto pobj = create_obj_bool();
            pobj->init = true;
            switch (val->ty)
            {
            case ObjType::INT:
                pobj->data.val = data.val == static_cast<const ObjInt*>(val.get())->data.val;
                return pobj;
            case ObjType::FLOAT:
                pobj->data.val = data.val == static_cast<const ObjFloat*>(val.get())->data.val;
                return pobj;
            }
            throw GenerationError("Expected int or float, recieved " + obj_type_name(val->ty));
        }

        template<>
        std::shared_ptr<Obj> ObjInt::ne(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);

            auto pobj = create_obj_bool();
            pobj->init = true;
            switch (val->ty)
            {
            case ObjType::INT:
                pobj->data.val = data.val != static_cast<const ObjInt*>(val.get())->data.val;
                return pobj;
            case ObjType::FLOAT:
                pobj->data.val = data.val != static_cast<const ObjFloat*>(val.get())->data.val;
                return pobj;
            }
            throw GenerationError("Expected int or float, recieved " + obj_type_name(val->ty));
        }

        template<>
        std::shared_ptr<Obj> ObjInt::ge(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);

            auto pobj = create_obj_bool();
            pobj->init = true;
            switch (val->ty)
            {
            case ObjType::INT:
                pobj->data.val = data.val >= static_cast<const ObjInt*>(val.get())->data.val;
                return pobj;
            case ObjType::FLOAT:
                pobj->data.val = data.val >= static_cast<const ObjFloat*>(val.get())->data.val;
                return pobj;
            }
            throw GenerationError("Expected int or float, recieved " + obj_type_name(val->ty));
        }

        template<>
        std::shared_ptr<Obj> ObjInt::le(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);

            auto pobj = create_obj_bool();
            pobj->init = true;
            switch (val->ty)
            {
            case ObjType::INT:
                pobj->data.val = data.val <= static_cast<const ObjInt*>(val.get())->data.val;
                return pobj;
            case ObjType::FLOAT:
                pobj->data.val = data.val <= static_cast<const ObjFloat*>(val.get())->data.val;
                return pobj;
            }
            throw GenerationError("Expected int or float, recieved " + obj_type_name(val->ty));
        }

        template<>
        std::shared_ptr<Obj> ObjInt::gt(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);

            auto pobj = create_obj_bool();
            pobj->init = true;
            switch (val->ty)
            {
            case ObjType::INT:
                pobj->data.val = data.val > static_cast<const ObjInt*>(val.get())->data.val;
                return pobj;
            case ObjType::FLOAT:
                pobj->data.val = data.val > static_cast<const ObjFloat*>(val.get())->data.val;
                return pobj;
            }
            throw GenerationError("Expected int or float, recieved " + obj_type_name(val->ty));
        }

        template<>
        std::shared_ptr<Obj> ObjInt::lt(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);

            auto pobj = create_obj_bool();
            pobj->init = true;
            switch (val->ty)
            {
            case ObjType::INT:
                pobj->data.val = data.val < static_cast<const ObjInt*>(val.get())->data.val;
                return pobj;
            case ObjType::FLOAT:
                pobj->data.val = data.val < static_cast<const ObjFloat*>(val.get())->data.val;
                return pobj;
            }
            throw GenerationError("Expected int or float, recieved " + obj_type_name(val->ty));
        }

        template<>
        ObjFloat::ObjImp() :
            Obj(ObjType::FLOAT)
        {
            data.val = 1.0;
        }

        template<>
        ObjFloat::~ObjImp() {}
        
        template<>
        std::shared_ptr<Obj> ObjFloat::copy() const
        {
            check_init(this);
            return create_obj_float(data.val);
        }

        template<>
        std::shared_ptr<Obj> ObjFloat::neg() const
        {
            check_init(this);
            auto pobj = create_obj_float();
            pobj->data.val = -data.val;
            pobj->init = true;
            return pobj;
        }

        template<>
        std::shared_ptr<Obj> ObjFloat::add(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);

            auto pobj = create_obj_float();
            if (val->ty == ObjType::INT)
                pobj->data.val = data.val + static_cast<const ObjInt*>(val.get())->data.val;
            else if (val->ty == ObjType::FLOAT)
                pobj->data.val = data.val + static_cast<const ObjFloat*>(val.get())->data.val;
            else
                throw GenerationError("Expected int or float, recieved " + obj_type_name(val->ty));

            pobj->init = true;
            return pobj;
        }

        template<>
        std::shared_ptr<Obj> ObjFloat::sub(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);

            auto pobj = create_obj_float();
            if (val->ty == ObjType::INT)
                pobj->data.val = data.val - static_cast<const ObjInt*>(val.get())->data.val;
            else if (val->ty == ObjType::FLOAT)
                pobj->data.val = data.val - static_cast<const ObjFloat*>(val.get())->data.val;
            else
                throw GenerationError("Expected int or float, recieved " + obj_type_name(val->ty));

            pobj->init = true;
            return pobj;
        }

        template<>
        std::shared_ptr<Obj> ObjFloat::mul(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);

            auto pobj = create_obj_float();
            if (val->ty == ObjType::INT)
                pobj->data.val = data.val * static_cast<const ObjInt*>(val.get())->data.val;
            else if (val->ty == ObjType::FLOAT)
                pobj->data.val = data.val * static_cast<const ObjFloat*>(val.get())->data.val;
            else
                throw GenerationError("Expected int or float, recieved " + obj_type_name(val->ty));

            pobj->init = true;
            return pobj;
        }

        template<>
        std::shared_ptr<Obj> ObjFloat::div(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);

            auto pobj = create_obj_float();
            if (val->ty == ObjType::INT)
                pobj->data.val = data.val / static_cast<const ObjInt*>(val.get())->data.val;
            else if (val->ty == ObjType::FLOAT)
                pobj->data.val = data.val / static_cast<const ObjFloat*>(val.get())->data.val;
            else
                throw GenerationError("Expected int or float, recieved " + obj_type_name(val->ty));

            pobj->init = true;
            return pobj;
        }

        template<>
        std::shared_ptr<Obj> ObjFloat::eq(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);

            auto pobj = create_obj_bool();
            if (val->ty == ObjType::INT)
                pobj->data.val = data.val == static_cast<const ObjInt*>(val.get())->data.val;
            else if (val->ty == ObjType::FLOAT)
                pobj->data.val = data.val == static_cast<const ObjFloat*>(val.get())->data.val;
            else
                throw GenerationError("Expected int or float, recieved " + obj_type_name(val->ty));

            pobj->init = true;
            return pobj;
        }

        template<>
        std::shared_ptr<Obj> ObjFloat::ne(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);

            auto pobj = create_obj_bool();
            if (val->ty == ObjType::INT)
                pobj->data.val = data.val != static_cast<const ObjInt*>(val.get())->data.val;
            else if (val->ty == ObjType::FLOAT)
                pobj->data.val = data.val != static_cast<const ObjFloat*>(val.get())->data.val;
            else
                throw GenerationError("Expected int or float, recieved " + obj_type_name(val->ty));

            pobj->init = true;
            return pobj;
        }

        template<>
        std::shared_ptr<Obj> ObjFloat::ge(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);

            auto pobj = create_obj_bool();
            if (val->ty == ObjType::INT)
                pobj->data.val = data.val >= static_cast<const ObjInt*>(val.get())->data.val;
            else if (val->ty == ObjType::FLOAT)
                pobj->data.val = data.val >= static_cast<const ObjFloat*>(val.get())->data.val;
            else
                throw GenerationError("Expected int or float, recieved " + obj_type_name(val->ty));

            pobj->init = true;
            return pobj;
        }

        template<>
        std::shared_ptr<Obj> ObjFloat::le(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);

            auto pobj = create_obj_bool();
            if (val->ty == ObjType::INT)
                pobj->data.val = data.val <= static_cast<const ObjInt*>(val.get())->data.val;
            else if (val->ty == ObjType::FLOAT)
                pobj->data.val = data.val <= static_cast<const ObjFloat*>(val.get())->data.val;
            else
                throw GenerationError("Expected int or float, recieved " + obj_type_name(val->ty));

            pobj->init = true;
            return pobj;
        }

        template<>
        std::shared_ptr<Obj> ObjFloat::gt(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);

            auto pobj = create_obj_bool();
            if (val->ty == ObjType::INT)
                pobj->data.val = data.val > static_cast<const ObjInt*>(val.get())->data.val;
            else if (val->ty == ObjType::FLOAT)
                pobj->data.val = data.val > static_cast<const ObjFloat*>(val.get())->data.val;
            else
                throw GenerationError("Expected int or float, recieved " + obj_type_name(val->ty));

            pobj->init = true;
            return pobj;
        }

        template<>
        std::shared_ptr<Obj> ObjFloat::lt(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);

            auto pobj = create_obj_bool();
            if (val->ty == ObjType::INT)
                pobj->data.val = data.val < static_cast<const ObjInt*>(val.get())->data.val;
            else if (val->ty == ObjType::FLOAT)
                pobj->data.val = data.val < static_cast<const ObjFloat*>(val.get())->data.val;
            else
                throw GenerationError("Expected int or float, recieved " + obj_type_name(val->ty));

            pobj->init = true;
            return pobj;
        }

        template<>
        ObjStr::ObjImp() :
            Obj(ObjType::STR)
        {
            data.val = "";
        }

        template<>
        ObjStr::~ObjImp() {}
        
        template<>
        void ObjStr::assign(const std::shared_ptr<Obj>& val)
        {
            check_init(val);
            check_mtype(val);

            data.val = mty(val)->data.val;
            init = true;
        }

        template<>
        std::shared_ptr<Obj> ObjStr::copy() const
        {
            check_init(this);
            return create_obj_str(data.val);
        }

        template<>
        std::string ObjStr::str() const
        {
            return data.val;
        }

        template<>
        std::shared_ptr<Obj> ObjStr::get(const std::string& item)
        {
            check_init(this);

            if (item == "length")
                return create_obj_int(data.val.size());

            throw GenerationError("str type has no member: " + item);
        }

        template<>
        std::vector<std::shared_ptr<Obj>> ObjStr::iter(EvalCtx& ctx)
        {
            check_init(this);

            std::vector<std::shared_ptr<Obj>> iters;
            for (auto e : data.val)
            {
                auto pelem = create_obj_str();
                pelem->data.val = std::to_string(e);
                pelem->init = true;
                iters.push_back(pelem);
            }
            return iters;
        }

        template<>
        std::shared_ptr<Obj> ObjStr::idx(const std::shared_ptr<Obj>& val)
        {
            check_init(this);
            check_init(val);
            if (val->ty != ObjType::INT)
                throw GenerationError("Expected int, recieved " + obj_type_name(val->ty));
            if (
                static_cast<const ObjInt*>(val.get())->data.val < 0 ||
                static_cast<const ObjInt*>(val.get())->data.val >= data.val.size())
                throw GenerationError("Index out of range");

            auto pobj = create_obj_str();
            pobj->data.val = data.val[static_cast<const ObjInt*>(val.get())->data.val];
            pobj->init = true;
            return pobj;
        }

        template<>
        std::shared_ptr<Obj> ObjStr::add(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);
            check_mtype(val);

            auto pobj = create_obj_str();
            pobj->data.val = data.val + mty(val)->data.val;
            pobj->init = true;
            return pobj;
        }

        template<>
        std::shared_ptr<Obj> ObjStr::eq(const std::shared_ptr<Obj>& val) const
        {
            // TODO: string comparison
            throw GenerationError("Not implemented");
        }

        template<>
        std::shared_ptr<Obj> ObjStr::ne(const std::shared_ptr<Obj>& val) const
        {
            // TODO: string comparison
            throw GenerationError("Not implemented");
        }

        template<>
        ObjArray::ObjImp() :
            Obj(ObjType::ARRAY)
        {
            data.dtype = nullptr;
            data.elems = {};
        }

        template<>
        ObjArray::~ObjImp() {}
        
        template<>
        void ObjArray::assign(const std::shared_ptr<Obj>& val)
        {
            check_init(val);
            check_mtype(val);

            if (init)
            {
                assert(data.elems.size() != -1);
                if (data.elems.size() != mty(val)->data.elems.size())
                    throw GenerationError("Width mismatch in array assignment");
                if (data.dtype->ne(mty(val)->data.dtype)->bval())
                    throw GenerationError("Type mismatch in array assignment");
                for (int i = 0; i < data.elems.size(); i++)
                {
                    check_init(mty(val)->data.elems[i]);
                    data.elems[i]->assign(mty(val)->data.elems[i]);
                }
            }
            else
            {
                data.dtype = mty(val)->data.dtype;
                for (auto& e : mty(val)->data.elems)
                {
                    check_init(e);
                    data.elems.push_back(e);
                }
                init = true;
            }
        }

        template<>
        std::shared_ptr<Obj> ObjArray::copy() const
        {
            check_init(this);
            std::vector<std::shared_ptr<Obj>> dups;
            for (auto e : data.elems)
                dups.push_back(e->copy());
            return create_obj_array(data.dtype, dups);
        }

        template<>
        std::shared_ptr<Obj> ObjArray::type() const
        {
            auto pobj = create_obj_dtype(ty);
            if (!init)
                return pobj;
            return pobj->cargs(std::static_pointer_cast<ObjDType>(data.dtype)->data.cargs);
        }

        template<>
        std::shared_ptr<Obj> ObjArray::get(const std::string& item)
        {
            check_init(this);

            if (item == "length")
                return create_obj_int(data.elems.size());

            throw GenerationError("array type has no member: " + item);
        }

        template<>
        std::vector<std::shared_ptr<Obj>> ObjArray::iter(EvalCtx& ctx)
        {
            check_init(this);
            for (auto e : data.elems)
                check_init(e);
            return data.elems;
        }

        template<>
        std::shared_ptr<Obj> ObjArray::idx(const std::shared_ptr<Obj>& val)
        {
            check_init(this);
            check_init(val);
            check_type(ObjType::INT, val);

            int64_t i = static_cast<const ObjInt*>(val.get())->data.val;
            if (i < 0 || data.elems.size() <= i)
                throw GenerationError("Index out of range");

            return data.elems[i];
        }

        template<>
        std::shared_ptr<Obj> ObjArray::add(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);
            check_mtype(val);
            if (data.dtype->ne(mty(val)->data.dtype)->bval())
                throw GenerationError("Array element type mismatch for concatenation");

            // concatenating the elements
            std::vector<std::shared_ptr<Obj>> nelems;
            for (auto e : data.elems)
                nelems.push_back(e->copy());
            for (auto e : mty(val)->data.elems)
                nelems.push_back(e->copy());
            return create_obj_array(data.dtype, nelems);
        }

        template<>
        std::shared_ptr<Obj> ObjArray::eq(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);
            check_mtype(val);

            if (data.elems.size() != mty(val)->data.elems.size())
                return create_obj_bool(false);

            for (int i = 0; i < data.elems.size(); i++)
                if (data.elems[i]->ne(mty(val)->data.elems[i])->bval())
                    return create_obj_bool(false);
            return create_obj_bool(true);
        }

        template<>
        std::shared_ptr<Obj> ObjArray::ne(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);
            check_mtype(val);

            if (data.elems.size() != mty(val)->data.elems.size())
                return create_obj_bool(true);

            for (int i = 0; i < data.elems.size(); i++)
                if (data.elems[i]->ne(mty(val)->data.elems[i])->bval())
                    return create_obj_bool(true);
            return create_obj_bool(false);
        }

        template<>
        ObjTuple::ObjImp() :
            Obj(ObjType::TUPLE)
        {
            data.dtypes = {};
            data.elems = {};
        }

        template<>
        ObjTuple::~ObjImp() {}
        
        template<>
        void ObjTuple::assign(const std::shared_ptr<Obj>& val)
        {
            check_init(val);
            check_mtype(val);
            if (init)
            {
                if (data.elems.size() != mty(val)->data.elems.size())
                    throw GenerationError("Width mismatch in tuple assignment");
                for (int i = 0; i < data.elems.size(); i++)
                    data.elems[i]->assign(mty(val)->data.elems[i]);
            }
            else
            {
                assert(data.elems.size() == 0);
                assert(data.dtypes.size() == 0);
                for (auto e : mty(val)->data.elems)
                    data.elems.push_back(e->copy());
                for (auto e : mty(val)->data.dtypes)
                    data.elems.push_back(e);
                init = true;
            }
        }

        template<>
        std::shared_ptr<Obj> ObjTuple::get(const std::string& item)
        {
            check_init(this);

            if (item == "length")
                return create_obj_int(data.elems.size());

            throw GenerationError("tuple type has no member: " + item);
        }

        template<>
        std::vector<std::shared_ptr<Obj>> ObjTuple::iter(EvalCtx& ctx)
        {
            check_init(this);
            for (auto e : data.elems)
                check_init(e);
            return data.elems;
        }

        template<>
        std::shared_ptr<Obj> ObjTuple::idx(const std::shared_ptr<Obj>& val)
        {
            check_init(this);
            check_init(val);
            if (val->ty != ObjType::INT)
                throw GenerationError("Expected int, recieved " + obj_type_name(val->ty));

            int64_t i = static_cast<const ObjInt*>(val.get())->data.val;
            if (i < 0 || data.elems.size() <= i)
                throw GenerationError("Index out of range");

            return data.elems[i];
        }

        template<>
        std::shared_ptr<Obj> ObjTuple::add(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);
            check_mtype(val);

            auto pobj = create_obj_tuple();
            for (auto e : data.elems)
                pobj->data.elems.push_back(e);
            for (auto e : mty(val)->data.elems)
                pobj->data.elems.push_back(e);
            pobj->init = true;
            return pobj;
        }

        template<>
        std::shared_ptr<Obj> ObjTuple::eq(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);
            check_mtype(val);

            if (data.elems.size() != mty(val)->data.elems.size())
                return create_obj_bool(false);

            for (int i = 0; i < data.elems.size(); i++)
                if (data.elems[i]->ne(mty(val)->data.elems[i]))
                    return create_obj_bool(false);
            return create_obj_bool(true);
        }

        template<>
        std::shared_ptr<Obj> ObjTuple::ne(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);
            check_mtype(val);

            if (data.elems.size() != mty(val)->data.elems.size())
                return create_obj_bool(true);

            for (int i = 0; i < data.elems.size(); i++)
                if (data.elems[i]->ne(mty(val)->data.elems[i]))
                    return create_obj_bool(true);
            return create_obj_bool(false);
        }

        template<>
        ObjTensor::ObjImp() :
            Obj(ObjType::TENSOR)
        {
            init = false;
            data.dims = {};
            data.pEdge = nullptr;
            data.carg_init = false;
            data.is_static = false;
        }

        template<>
        ObjTensor::~ObjImp() {}

        template<>
        void ObjTensor::assign(const std::shared_ptr<Obj>& val)
        {
            check_mtype(val);
            check_init(val);

            // TODO: handle assignment to static tensors

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

        std::shared_ptr<Obj> ObjTensor::copy() const
        {
            check_init(this);
            assert(data.carg_init);

            auto pobj = std::make_shared<ObjTensor>();
            pobj->data.pEdge = data.pEdge;
            pobj->data.dims = data.dims;  // copy assignment
            pobj->data.carg_init = true;
            pobj->init = true;
            return pobj;
        }

        template<>
        std::vector<std::shared_ptr<Obj>> ObjTensor::iter(EvalCtx& ctx)
        {
            std::vector<std::shared_ptr<Obj>> iters;
            for (auto e : data.dims)
                iters.push_back(create_obj_int(e));
            return iters;
        }

        std::shared_ptr<Obj> ObjTensor::eq(const std::shared_ptr<Obj>& val) const
        {
            // TODO: check if the tensor rank and dimensions are equal (required for carg deduction)
            throw GenerationError("Not implemented");
        }

        std::shared_ptr<Obj> ObjTensor::ne(const std::shared_ptr<Obj>& val) const
        {
            // TODO: check if the tensor rank or dimensions are not equal (required for carg deduction)
            throw GenerationError("Not implemented");
        }

        template<>
        ObjDef::ObjImp() :
            Obj(ObjType::DEF)
        {
            data.pdef = nullptr;
            data.cargs = {};
            data.has_cargs = false;
        }

        template<>
        ObjDef::~ObjImp() {}

        template<>
        std::shared_ptr<Obj> ObjDef::cargs(const std::vector<std::shared_ptr<Obj>>& args)
        {
            check_init(this);
            if (data.has_cargs)
                throw GenerationError("Const args have already been set");
            return create_obj_def(data.pdef, args);
        }

        template<>
        void ObjDef::call(EvalCtx& ctx, const std::vector<std::shared_ptr<Obj>>& args) const
        {
            check_init(this);
            for (auto e : args)
                check_init(e);

            // creating a new scope for the def call
            Scope* pscope = new Scope();

            // saving the old state
            Scope* prev_scope = ctx.pscope;
            EvalState prev_state = ctx.state;
            std::string prev_block = ctx.block_name;

            // creating the new state
            ctx.pscope = pscope;
            ctx.state = EvalState::DEFSEQ;
            ctx.block_name = data.pdef->get_name();

            // applying the cargs, and evaluating the args
            data.pdef->apply_cargs(ctx, data.cargs);  // If its empty, its empty
            std::vector<std::shared_ptr<Obj>> cpy_args;
            if (prev_state == EvalState::STARTUP)
            {
                // Generating the arguments for the entry point
                for (const auto& [decl, name] : data.pdef->get_vargs())
                    cpy_args.push_back(decl.auto_gen(ctx, name));  // automatically generating the arguments one by one
            }
            else
            {
                // Normal def call, just copy the arguments
                for (auto e : args)
                    cpy_args.push_back(e->copy());
            }

            // doing the call
            data.pdef->carg_deduction(ctx, cpy_args);
            // ensuring the variables in the scope are fully initialized
            for (auto e : ctx.scope())
                check_init(std::get<1>(e));
            auto pret = data.pdef->get_body().eval(ctx);  // not actually the return value, that's in last_ret
            assert(pret->ty == ObjType::INVALID);

            // restoring the previous state and cleanup
            ctx.pscope = prev_scope;
            ctx.state = prev_state;
            ctx.block_name = prev_block;
            delete pscope;
        }

        template<>
        ObjFn::ObjImp() :
            Obj(ObjType::FN)
        {
            // TODO: Implement functions
            throw GenerationError("Not Implemented");
        }

        template<>
        ObjFn::~ObjImp()
        {
            // TODO: Implement functions
            throw GenerationError("Not Implemented");
        }

        template<>
        ObjIntr::ObjImp() :
            Obj(ObjType::INTR)
        {
            data.pintr = nullptr;
            data.cargs = {};
            data.has_cargs = false;
        }

        template<>
        ObjIntr::~ObjImp() {}

        template<>
        std::shared_ptr<Obj> ObjIntr::cargs(const std::vector<std::shared_ptr<Obj>>& args)
        {
            check_init(this);
            if (data.has_cargs)
                throw GenerationError("Const args have already been set");
            return create_obj_intr(data.pintr, args);
        }

        template<>
        void ObjIntr::call(EvalCtx& ctx, const std::vector<std::shared_ptr<Obj>>& args) const
        {
            check_init(this);
            for (auto e : args)
                check_init(e);

            // copying the arguments for the new scope
            std::vector<std::shared_ptr<Obj>> cpy_args;
            for (auto e : args)
            {
                check_type(ObjType::TENSOR, e);
                assert(std::static_pointer_cast<ObjTensor>(e)->data.pEdge);
                cpy_args.push_back(e->copy());
            }

            // creating a new scope for the intr call
            Scope* pscope = new Scope();

            // saving the old state
            Scope* prev_scope = ctx.pscope;
            EvalState prev_state = ctx.state;
            std::string prev_block = ctx.block_name;

            // creating the new state
            ctx.pscope = pscope;
            ctx.state = EvalState::INTR;
            ctx.block_name = data.pintr->get_name();

            // applying the cargs, and evaluating the args
            data.pintr->apply_cargs(ctx, data.cargs);  // If its empty, its empty
            data.pintr->carg_deduction(ctx, cpy_args);
            // ensuring the variables in the scope are fully initialized
            for (auto e : ctx.scope())
                check_init(std::get<1>(e));
            auto pret = data.pintr->get_body().eval(ctx);  // not actually the return value, that's in last_ret
            assert(pret->ty == ObjType::INVALID);

            // restoring the previous state and cleanup
            ctx.pscope = prev_scope;
            ctx.state = prev_state;
            ctx.block_name = prev_block;
            delete pscope;

            // creating the new node based on the inputs and output from the intrinsic
            core::Node* pnode = new core::Node();
            pnode->name = data.pintr->get_name();
            for (auto e : data.cargs)
                pnode->cargs.push_back(e->copy());
            // node input connections
            for (int i = 0; i < args.size(); i++)
            {
                std::static_pointer_cast<ObjTensor>(args[i])->data.pEdge->outputs.push_back({ pnode, i });  // edge output to node input
                pnode->inputs.push_back(std::static_pointer_cast<ObjTensor>(args[i])->data.pEdge);  // node input to edge output
            }
            // node output connections
            if (last_ret->ty == ObjType::TENSOR)
            {
                // single output

                ObjTensor* pten = static_cast<ObjTensor*>(last_ret.get());
                if (!pten->data.pEdge)  // creating the edge if needed
                {
                    if (!pten->data.carg_init)
                        throw GenerationError("Intrinsic output tensors must have a known shape");
                    // creating a new edge
                    pten->data.pEdge = new core::Edge();
                    pten->data.pEdge->dsc.rk = pten->data.dims.size();
                    for (auto e : pten->data.dims)
                        pten->data.pEdge->dsc.dims.push_back(e);
                }
                else if (pten->data.pEdge->input)  // if the edge already exists, make sure the input hasn't been mapped yet
                    throw GenerationError("Edge input has already been mapped");
                assert(pten->data.pEdge->inpid == -1);

                pten->data.pEdge->input = pnode;  // edge input to node output
                pten->data.pEdge->inpid = 0;
                pnode->outputs.push_back(pten->data.pEdge);  // node output to edge input
            }
            else
            {
                // multiple outputs
                const auto& outs = last_ret->iter(ctx);
                for (int i = 0; i < outs.size(); i++)
                {
                    check_type(ObjType::TENSOR, outs[i]);

                    ObjTensor* pten = static_cast<ObjTensor*>(outs[i].get());
                    if (!pten->data.pEdge)  // creating the edge if needed
                    {
                        if (!pten->data.carg_init)
                            throw GenerationError("Intrinsic output tensors must have a known shape");
                        // creating a new edge
                        pten->data.pEdge = new core::Edge();
                        pten->data.pEdge->dsc.rk = pten->data.dims.size();
                        for (auto e : pten->data.dims)
                            pten->data.pEdge->dsc.dims.push_back(e);
                    }
                    else if (pten->data.pEdge->input)  // if the edge already exists, make sure the input hasn't been mapped yet
                        throw GenerationError("Edge input has already been mapped");
                    assert(pten->data.pEdge->inpid == -1);

                    pten->data.pEdge->input = pnode;  // edge input to node output
                    pten->data.pEdge->inpid = i;
                    pnode->outputs.push_back(pten->data.pEdge);  // node output to edge input
                }
            }
        }

        template<>
        ObjModule::ObjImp() :
            Obj(ObjType::MODULE)
        {
            // TODO: Implement modules
            throw GenerationError("Not Implmented");
        }

        ObjModule::~ObjImp()
        {
            // TODO: Implement modules
            throw GenerationError("Not Implmented");
        }

        template<>
        ObjPackage::ObjImp() :
            Obj(ObjType::PACKAGE)
        {
            // TODO: Implement packages
            throw GenerationError("Not Implmented");
        }

        template<>
        ObjPackage::~ObjImp()
        {
            // TODO: Implement packages
            throw GenerationError("Not Implmented");
        }

        // Object contructors, not going to use factories

        std::shared_ptr<Obj> create_obj_type(ObjType ty)
        {
            std::shared_ptr<Obj> pelem;
            switch (ty)
            {
            case ObjType::TYPE   :    pelem = create_obj_dtype   (); break;
            case ObjType::VAR    :    pelem = create_obj_var     (); break;
            case ObjType::BOOL   :    pelem = create_obj_bool    (); break;
            case ObjType::INT    :    pelem = create_obj_int     (); break;
            case ObjType::FLOAT  :    pelem = create_obj_float   (); break;
            case ObjType::STR    :    pelem = create_obj_str     (); break;
            case ObjType::ARRAY  :    pelem = create_obj_array   (); break;
            case ObjType::TUPLE  :    pelem = create_obj_tuple   (); break;
            case ObjType::TENSOR :    pelem = create_obj_tensor  (); break;
            case ObjType::DEF    :    pelem = create_obj_def     (); break;
            case ObjType::FN     :    pelem = create_obj_fn      (); break;
            case ObjType::INTR   :    pelem = create_obj_intr    (); break;
            case ObjType::MODULE :    pelem = create_obj_module  (); break;
            case ObjType::PACKAGE:    pelem = create_obj_package (); break;
            default:
                throw GenerationError("Invalid object type for contruction: " + obj_type_name(ty));
            }
            return pelem;
        }

        std::shared_ptr<ObjDType> create_obj_dtype()
        {
            auto pobj = std::make_shared<ObjDType>();
            pobj->init = false;
            return pobj;
        }

        std::shared_ptr<ObjDType> create_obj_dtype(ObjType ty)
        {
            auto pobj = std::make_shared<ObjDType>();
            pobj->data.ety = ty;
            pobj->data.cargs = {};
            pobj->data.has_cargs = false;
            pobj->init = true;
            return pobj;
        }

        std::shared_ptr<ObjDType> create_obj_dtype(ObjType ty, const std::vector<std::shared_ptr<Obj>>& cargs)
        {
            auto pobj = std::make_shared<ObjDType>();
            pobj->data.ety = ty;
            for (auto e : cargs)
                check_init(e);
            pobj->data.cargs = cargs;
            pobj->data.has_cargs = true;
            pobj->init = true;
            return pobj;
        }

        std::shared_ptr<ObjInvalid> create_obj_invalid()
        {
            return std::make_shared<ObjInvalid>();
        }

        std::shared_ptr<ObjVar> create_obj_var()
        {
            return std::make_shared<ObjVar>();
        }

        std::shared_ptr<ObjFWidth> create_obj_fwidth()
        {
            return std::make_shared<ObjFWidth>();
        }

        std::shared_ptr<ObjFWidth> create_obj_fwidth(core::tensor_dty dty)
        {
            auto pobj = std::make_shared<ObjFWidth>();
            pobj->data.dty = dty;
            pobj->init = true;
            return pobj;
        }

        std::shared_ptr<ObjBool> create_obj_bool()
        {
            auto pobj = std::make_shared<ObjBool>();
            pobj->init = false;
            return pobj;
        }

        std::shared_ptr<ObjBool> create_obj_bool(bool val)
        {
            auto pobj = std::make_shared<ObjBool>();
            pobj->data.val = val;
            pobj->init = true;
            return pobj;
        }

        std::shared_ptr<ObjInt> create_obj_int()
        {
            auto pobj = std::make_shared<ObjInt>();
            pobj->init = false;
            return pobj;
        }

        std::shared_ptr<ObjInt> create_obj_int(int64_t val)
        {
            auto pobj = std::make_shared<ObjInt>();
            pobj->data.val = val;
            pobj->init = true;
            return pobj;
        }

        std::shared_ptr<ObjFloat> create_obj_float()
        {
            auto pobj = std::make_shared<ObjFloat>();
            pobj->init = false;
            return pobj;
        }

        std::shared_ptr<ObjFloat> create_obj_float(double val)
        {
            auto pobj = std::make_shared<ObjFloat>();
            pobj->data.val = val;
            pobj->init = true;
            return pobj;
        }

        std::shared_ptr<ObjStr> create_obj_str()
        {
            auto pobj = std::make_shared<ObjStr>();
            pobj->init = false;
            return pobj;
        }

        std::shared_ptr<ObjStr> create_obj_str(const std::string& val)
        {
            auto pobj = std::make_shared<ObjStr>();
            pobj->data.val = val;
            pobj->init = true;
            return pobj;
        }

        std::shared_ptr<ObjArray> create_obj_array()
        {
            return std::make_shared<ObjArray>();
        }

        std::shared_ptr<ObjArray> create_obj_array(const std::shared_ptr<Obj> dtype, int sz)
        {
            auto pobj = std::make_shared<ObjArray>();
            pobj->data.dtype = dtype;
            for (int i = 0; i < sz; i++)
                pobj->data.elems.push_back(dtype->inst());
            pobj->init = true;
            return pobj;
        }

        std::shared_ptr<ObjArray> create_obj_array(const std::shared_ptr<Obj> dtype, const std::vector<std::shared_ptr<Obj>>& elems)
        {
            auto pobj = std::make_shared<ObjArray>();
            pobj->data.dtype = dtype;
            for (auto e : elems)
            {
                pobj->data.elems.push_back(dtype->inst());
                pobj->data.elems.back()->assign(e);
            }
            pobj->init = true;
            return pobj;
        }

        std::shared_ptr<ObjTuple> create_obj_tuple()
        {
            return std::make_shared<ObjTuple>();
        }

        std::shared_ptr<ObjTuple> create_obj_tuple(const std::vector<std::shared_ptr<ObjDType>>& dtypes)
        {
            auto pobj = std::make_shared<ObjTuple>();
            for (auto e : dtypes)
            {
                pobj->data.dtypes.push_back(e);
                pobj->data.elems.push_back(e->inst());
            }
            pobj->init = true;
            return pobj;
        }

        std::shared_ptr<ObjTuple> create_obj_tuple(const std::vector<std::shared_ptr<Obj>>& elems)
        {
            auto pobj = std::make_shared<ObjTuple>();
            pobj->data.elems = elems;
            for (auto e : elems)
                pobj->data.dtypes.push_back(e->type());
            pobj->init = true;
            return pobj;
        }

        std::shared_ptr<ObjTensor> create_obj_tensor()
        {
            auto pobj = std::make_shared<ObjTensor>();
            pobj->data.pEdge = nullptr;
            pobj->data.dims = {};
            pobj->data.carg_init = false;
            return pobj;
        }

        std::shared_ptr<ObjTensor> create_obj_tensor(core::tensor_dty dty, const std::vector<uint32_t>& dims)
        {
            auto pobj = std::make_shared<ObjTensor>();
            pobj->data.pEdge = nullptr;
            pobj->data.dims = dims;
            pobj->data.dty = dty;
            pobj->data.carg_init = true;
            pobj->init = true;
            return pobj;
        }

        std::shared_ptr<ObjDef> create_obj_def()
        {
            return std::make_shared<ObjDef>();
        }

        std::shared_ptr<ObjDef> create_obj_def(const AstDef* pdef)
        {
            auto pobj = std::make_shared<ObjDef>();
            pobj->data.pdef = pdef;
            pobj->init = true;
            return pobj;
        }

        std::shared_ptr<ObjDef> create_obj_def(const AstDef* pdef, const std::vector<std::shared_ptr<Obj>>& cargs)
        {
            auto pobj = std::make_shared<ObjDef>();
            pobj->data.pdef = pdef;
            for (auto e : cargs)
                check_init(e);
            pobj->data.cargs = cargs;
            pobj->data.has_cargs = true;
            pobj->init = true;
            return pobj;
        }

        std::shared_ptr<ObjFn> create_obj_fn()
        {
            return std::make_shared<ObjFn>();
        }

        std::shared_ptr<ObjFn> create_obj_fn(const AstFn* pfn)
        {
            auto pobj = std::make_shared<ObjFn>();
            pobj->data.pfn = pfn;
            pobj->init = true;
            return pobj;
        }

        std::shared_ptr<ObjIntr> create_obj_intr()
        {
            return std::make_shared<ObjIntr>();
        }

        std::shared_ptr<ObjIntr> create_obj_intr(const AstIntr* pintr)
        {
            auto pobj = std::make_shared<ObjIntr>();
            pobj->data.pintr = pintr;
            pobj->init = true;
            return pobj;
        }

        std::shared_ptr<ObjIntr> create_obj_intr(const AstIntr* pintr, const std::vector<std::shared_ptr<Obj>>& cargs)
        {
            auto pobj = std::make_shared<ObjIntr>();
            pobj->data.pintr = pintr;
            for (auto e : cargs)
                check_init(e);
            pobj->data.cargs = cargs;
            pobj->data.has_cargs = true;
            pobj->init = true;
            return pobj;
        }

        std::shared_ptr<ObjModule> create_obj_module()
        {
            throw GenerationError("Not implemented");
        }

        std::shared_ptr<ObjPackage> create_obj_package()
        {
            throw GenerationError("Not Implemented");
        }
    }
}
