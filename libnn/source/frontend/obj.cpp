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

        void check_init(const std::shared_ptr<Obj>& pobj)
        {
            if (!pobj->init)
                throw GenerationError("Uninitialized variable");
        }

        constexpr std::string obj_type_name(ObjType ty)
        {
            switch (ty)
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
        template<> ObjArray::~ObjImp() {}
        template<> ObjTuple::~ObjImp() {}
        template<> ObjTensor::~ObjImp() {}

        template<>
        ObjBool::ObjImp() :
            Obj(ObjType::BOOL)
        {
            data.val = false;
        }

        template<>
        bool ObjBool::bval() const
        {
            check_init(this);
            return data.val;
        }

        template<>
        void ObjBool::assign(const std::shared_ptr<Obj>& val)
        {
            check_type(val);
            check_init(val);
            data.val = mty(val)->data.val;
            init = true;
        }

        template<>
        std::shared_ptr<Obj> ObjBool::andop(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);
            check_type(val);

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
            check_type(val);

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
            check_type(val);

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
            check_type(val);

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
        void ObjInt::assign(const std::shared_ptr<Obj>& val)
        {
            check_type(val);
            check_init(val);
            data.val = mty(val)->data.val;
            init = true;
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
        void ObjStr::assign(const std::shared_ptr<Obj>& val)
        {
            check_init(val);
            check_type(val);

            data.val = mty(val)->data.val;
            init = true;
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
            check_type(val);

            auto pobj = create_obj_str();
            pobj->data.val = data.val + mty(val)->data.val;
            pobj->init = true;
            return pobj;
        }

        template<>
        ObjArray::ObjImp() :
            Obj(ObjType::ARRAY)
        {
            data.size = -1;
            data.ety = ObjType::INVALID;
            data.elems = {};
        }

        template<>
        void ObjArray::assign(const std::shared_ptr<Obj>& val)
        {
            check_init(val);
            check_type(val);
            if (init)
            {
                assert(data.size != -1);
                if (data.size != mty(val)->data.size)
                    throw GenerationError("Width mismatch in array assignment");
                if (data.ety != mty(val)->data.ety)
                    throw GenerationError("Type mismatch in array assignment");
                for (int i = 0; i < data.size; i++)
                {
                    check_init(mty(val)->data.elems[i]);
                    data.elems[i] = mty(val)->data.elems[i];
                }
            }
            else
            {
                assert(data.elems.size() == 0);
                data.size = mty(val)->data.size;
                data.ety = mty(val)->data.ety;
                for (auto& e : mty(val)->data.elems)
                {
                    check_init(e);
                    data.elems.push_back(e);
                }
                init = true;
            }
        }

        template<>
        std::shared_ptr<Obj> ObjArray::get(const std::string& item)
        {
            check_init(this);

            if (item == "length")
                return create_obj_int(data.size);
            if (item == "dtype")
                return create_obj_gentype(data.ety);

            throw GenerationError("array type has no member: " + item);
        }

        template<>
        std::shared_ptr<Obj> ObjArray::idx(const std::shared_ptr<Obj>& val)
        {
            check_init(this);
            check_init(val);
            if (val->ty != ObjType::INT)
                throw GenerationError("Expected int, recieved " + obj_type_name(val->ty));

            int64_t i = static_cast<const ObjInt*>(val.get())->data.val;
            if (i < 0 || data.size <= i)
                throw GenerationError("Index out of range");

            return data.elems[i];
        }

        template<>
        std::shared_ptr<Obj> ObjArray::add(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);
            check_type(val);
            if (data.ety != mty(val)->data.ety)
                throw GenerationError("Array element type mismatch for concatenation");

            auto pobj = create_obj_array(data.size + mty(val)->data.size, data.ety);
            for (int i = 0; i < data.size; i++)
                pobj->data.elems[i]->assign(data.elems[i]);
            for (int i = 0; i < data.size; i++)
                pobj->data.elems[i + data.size]->assign(mty(val)->data.elems[i]);

            return pobj;
        }

        template<>
        std::shared_ptr<Obj> ObjArray::eq(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);
            check_type(val);

            if (data.size != mty(val)->data.size)
                return create_obj_bool(false);

            for (int i = 0; i < data.size; i++)
                if (data.elems[i]->ne(mty(val)->data.elems[i]))
                    return create_obj_bool(false);
            return create_obj_bool(true);
        }

        template<>
        std::shared_ptr<Obj> ObjArray::ne(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);
            check_type(val);

            if (data.size != mty(val)->data.size)
                return create_obj_bool(true);
            
            for (int i = 0; i < data.size; i++)
                if (data.elems[i]->ne(mty(val)->data.elems[i]))
                    return create_obj_bool(true);
            return create_obj_bool(false);
        }

        template<>
        ObjTuple::ObjImp() :
            Obj(ObjType::TUPLE)
        {
            data.elems = {};
        }

        template<>
        void ObjTuple::assign(const std::shared_ptr<Obj>& val)
        {
            check_init(val);
            check_type(val);
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
                for (auto e : mty(val)->data.elems)
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
            check_type(val);

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
            check_type(val);

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
            check_type(val);

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
        }

        template<>
        void ObjTensor::assign(const std::shared_ptr<Obj>& val)
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
        std::shared_ptr<Obj> ObjTensor::add(const std::shared_ptr<Obj>& val) const
        {

        }

        template<>
        std::shared_ptr<Obj> ObjTensor::sub(const std::shared_ptr<Obj>& val) const
        {

        }

        template<>
        std::shared_ptr<Obj> ObjTensor::mul(const std::shared_ptr<Obj>& val) const
        {

        }

        template<>
        std::shared_ptr<Obj> ObjTensor::div(const std::shared_ptr<Obj>& val) const
        {

        }

        template<>
        ObjDef::ObjImp() :
            Obj(ObjType::DEF)
        {
            
        }

        // Object contructors, not going to use factories

        std::shared_ptr<Obj> create_obj_type(ObjType ty)
        {
            std::shared_ptr<Obj> pelem;
            switch (ty)
            {
            case ObjType::TYPE   :    pelem = create_obj_gentype (); break;
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
        
        std::shared_ptr<ObjInvalid> create_obj_invalid()
        {
            return std::make_shared<ObjInvalid>();
        }

        std::shared_ptr<ObjGenType> create_obj_gentype()
        {
            auto pobj = std::make_shared<ObjGenType>();
            pobj->init = false;
            return pobj;
        }

        std::shared_ptr<ObjGenType> create_obj_gentype(ObjType ty)
        {
            auto pobj = std::make_shared<ObjGenType>();
            pobj->data.val = ty;
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
            auto pobj = std::make_shared<ObjArray>();
            pobj->init = false;
            return pobj;
        }

        std::shared_ptr<ObjArray> create_obj_array(size_t sz, ObjType ty)
        {
            auto pobj = std::make_shared<ObjArray>();
            pobj->data.size = sz;
            for (int i = 0; i < sz; i++)
                pobj->data.elems.push_back(create_obj_type(ty));
            pobj->init = true;
            return pobj;
        }

        std::shared_ptr<ObjTuple> create_obj_tuple()
        {
            auto pobj = std::make_shared<ObjTuple>();
            pobj->init = false;
            return pobj;
        }

        std::shared_ptr<ObjTuple> create_obj_tuple(const std::vector<std::shared_ptr<Obj>>& elems)
        {
            auto pobj = std::make_shared<ObjTuple>();
            for (auto e : elems)
                pobj->data.elems.push_back(e);
            pobj->init = true;
            return pobj;
        }

        std::shared_ptr<ObjTensor> create_obj_tensor()
        {
            auto pobj = std::make_shared<ObjTensor>();
            pobj->data.pEdge = nullptr;
            pobj->data.dims = {};
            pobj->data.carg_init = false;
            pobj->init = false;
        }

        std::shared_ptr<ObjDef> create_obj_def()
        {

        }

        std::shared_ptr<ObjFn> create_obj_fn()
        {

        }

        std::shared_ptr<ObjIntr> create_obj_intr()
        {

        }
    }
}
