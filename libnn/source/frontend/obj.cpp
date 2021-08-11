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

            auto pobj = create_obj<ObjType::BOOL>();
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

            auto pobj = create_obj<ObjType::BOOL>();
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

            auto pobj = create_obj<ObjType::BOOL>();
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

            auto pobj = create_obj<ObjType::BOOL>();
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
            auto pobj = create_obj<ObjType::INT>();
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
                auto pobj = create_obj<ObjType::INT>();
                pobj->data.val = data.val + static_cast<const ObjInt*>(val.get())->data.val;
                pobj->init = true;
                return pobj;
            }
            else if (val->ty == ObjType::FLOAT)
            {
                auto pobj = create_obj<ObjType::FLOAT>();
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
                auto pobj = create_obj<ObjType::INT>();
                pobj->data.val = data.val - static_cast<const ObjInt*>(val.get())->data.val;
                pobj->init = true;
                return pobj;
            }
            else if (val->ty == ObjType::FLOAT)
            {
                auto pobj = create_obj<ObjType::FLOAT>();
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
                auto pobj = create_obj<ObjType::INT>();
                pobj->data.val = data.val * static_cast<const ObjInt*>(val.get())->data.val;
                pobj->init = true;
                return pobj;
            }
            else if (val->ty == ObjType::FLOAT)
            {
                auto pobj = create_obj<ObjType::FLOAT>();
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
                auto pobj = create_obj<ObjType::INT>();
                pobj->data.val = data.val / static_cast<const ObjInt*>(val.get())->data.val;
                pobj->init = true;
                return pobj;
            }
            else if (val->ty == ObjType::FLOAT)
            {
                auto pobj = create_obj<ObjType::FLOAT>();
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

            auto pobj = create_obj<ObjType::BOOL>();
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

            auto pobj = create_obj<ObjType::BOOL>();
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

            auto pobj = create_obj<ObjType::BOOL>();
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

            auto pobj = create_obj<ObjType::BOOL>();
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

            auto pobj = create_obj<ObjType::BOOL>();
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

            auto pobj = create_obj<ObjType::BOOL>();
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
            auto pobj = create_obj<ObjType::FLOAT>();
            pobj->data.val = -data.val;
            pobj->init = true;
            return pobj;
        }

        template<>
        std::shared_ptr<Obj> ObjFloat::add(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);

            auto pobj = create_obj<ObjType::FLOAT>();
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

            auto pobj = create_obj<ObjType::FLOAT>();
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

            auto pobj = create_obj<ObjType::FLOAT>();
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

            auto pobj = create_obj<ObjType::FLOAT>();
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

            auto pobj = create_obj<ObjType::BOOL>();
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

            auto pobj = create_obj<ObjType::BOOL>();
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

            auto pobj = create_obj<ObjType::BOOL>();
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

            auto pobj = create_obj<ObjType::BOOL>();
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

            auto pobj = create_obj<ObjType::BOOL>();
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

            auto pobj = create_obj<ObjType::BOOL>();
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
                return create_obj<ObjType::INT>(data.val.size());

            throw GenerationError("str type has no member: " + item);
        }

        template<>
        std::vector<std::shared_ptr<Obj>> ObjStr::iter(EvalCtx& ctx)
        {
            check_init(this);

            std::vector<std::shared_ptr<Obj>> iters;
            for (auto e : data.val)
            {
                auto pelem = create_obj<ObjType::STR>();
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

            auto pobj = create_obj<ObjType::STR>();
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

            auto pobj = create_obj<ObjType::STR>();
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
                return create_obj<ObjType::INT>(data.size);
            if (item == "dtype")
                return create_obj<ObjType::TYPE>(data.ety);

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

            auto pobj = create_obj<ObjType::ARRAY>(data.size + mty(val)->data.size, data.ety);
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
                return create_obj<ObjType::BOOL>(false);

            for (int i = 0; i < data.size; i++)
                if (data.elems[i]->ne(mty(val)->data.elems[i]))
                    return create_obj<ObjType::BOOL>(false);
            return create_obj<ObjType::BOOL>(true);
        }

        template<>
        std::shared_ptr<Obj> ObjArray::ne(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);
            check_type(val);

            if (data.size != mty(val)->data.size)
                return create_obj<ObjType::BOOL>(true);
            
            for (int i = 0; i < data.size; i++)
                if (data.elems[i]->ne(mty(val)->data.elems[i]))
                    return create_obj<ObjType::BOOL>(true);
            return create_obj<ObjType::BOOL>(false);
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
                return create_obj<ObjType::INT>(data.elems.size());

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

            auto pobj = create_obj<ObjType::TUPLE>();
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
                return create_obj<ObjType::BOOL>(false);

            for (int i = 0; i < data.elems.size(); i++)
                if (data.elems[i]->ne(mty(val)->data.elems[i]))
                    return create_obj<ObjType::BOOL>(false);
            return create_obj<ObjType::BOOL>(true);
        }

        template<>
        std::shared_ptr<Obj> ObjTuple::ne(const std::shared_ptr<Obj>& val) const
        {
            check_init(this);
            check_init(val);
            check_type(val);

            if (data.elems.size() != mty(val)->data.elems.size())
                return create_obj<ObjType::BOOL>(true);

            for (int i = 0; i < data.elems.size(); i++)
                if (data.elems[i]->ne(mty(val)->data.elems[i]))
                    return create_obj<ObjType::BOOL>(true);
            return create_obj<ObjType::BOOL>(false);
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

                // TODO: apply the cargs to the graph edge
            }

            // doing the actual assignment
            data.pEdge = mty(val)->data.pEdge;
            init = true;
        }

        template<>
        std::shared_ptr<Obj> ObjTensor::cargs(const std::vector<std::shared_ptr<Obj>>& cargs)
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

            // TODO: apply the cargs to the graph edge
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
            case ObjType::BOOL   :    pelem = create_obj<ObjType::BOOL   >(); break;
            case ObjType::INT    :    pelem = create_obj<ObjType::INT    >(); break;
            case ObjType::FLOAT  :    pelem = create_obj<ObjType::FLOAT  >(); break;
            case ObjType::STR    :    pelem = create_obj<ObjType::STR    >(); break;
            case ObjType::ARRAY  :    pelem = create_obj<ObjType::ARRAY  >(); break;
            case ObjType::TUPLE  :    pelem = create_obj<ObjType::TUPLE  >(); break;
            case ObjType::TENSOR :    pelem = create_obj<ObjType::TENSOR >(); break;
            case ObjType::DEF    :    pelem = create_obj<ObjType::DEF    >(); break;
            case ObjType::FN     :    pelem = create_obj<ObjType::FN     >(); break;
            case ObjType::INTR   :    pelem = create_obj<ObjType::INTR   >(); break;
            case ObjType::MODULE :    pelem = create_obj<ObjType::MODULE >(); break;
            case ObjType::PACKAGE:    pelem = create_obj<ObjType::PACKAGE>(); break;
            default:
                throw GenerationError("Invalid object type for contruction: " + obj_type_name(ty));
            }
            return pelem;
        }

        template<>
        std::shared_ptr<ObjInvalid> create_obj<ObjType::INVALID>()
        {
            return std::make_shared<ObjInvalid>();
        }

        template<>
        std::shared_ptr<ObjBool> create_obj<ObjType::BOOL>()
        {
            auto pobj = std::make_shared<ObjBool>();
            pobj->init = false;
            return pobj;
        }

        template<>
        std::shared_ptr<ObjBool> create_obj<ObjType::BOOL, bool>(bool val)
        {
            auto pobj = std::make_shared<ObjBool>();
            pobj->data.val = val;
            pobj->init = true;
            return pobj;
        }

        template<>
        std::shared_ptr<ObjInt> create_obj<ObjType::INT>()
        {
            auto pobj = std::make_shared<ObjInt>();
            pobj->init = false;
            return pobj;
        }

        template<>
        std::shared_ptr<ObjInt> create_obj<ObjType::INT, int64_t>(int64_t val)
        {
            auto pobj = std::make_shared<ObjInt>();
            pobj->data.val = val;
            pobj->init = true;
            return pobj;
        }

        template<>
        std::shared_ptr<ObjFloat> create_obj<ObjType::FLOAT>()
        {
            auto pobj = std::make_shared<ObjFloat>();
            pobj->init = false;
            return pobj;
        }

        template<>
        std::shared_ptr<ObjFloat> create_obj<ObjType::FLOAT, double>(double val)
        {
            auto pobj = std::make_shared<ObjFloat>();
            pobj->data.val = val;
            pobj->init = true;
            return pobj;
        }

        template<>
        std::shared_ptr<ObjStr> create_obj<ObjType::STR>()
        {
            auto pobj = std::make_shared<ObjStr>();
            pobj->init = false;
            return pobj;
        }

        template<>
        std::shared_ptr<ObjStr> create_obj<ObjType::STR, const std::string&>(const std::string& val)
        {
            auto pobj = std::make_shared<ObjStr>();
            pobj->data.val = val;
            pobj->init = true;
            return pobj;
        }

        template<>
        std::shared_ptr<ObjArray> create_obj<ObjType::ARRAY>()
        {
            auto pobj = std::make_shared<ObjArray>();
            pobj->init = false;
            return pobj;
        }

        template<>
        std::shared_ptr<ObjArray> create_obj<ObjType::ARRAY, size_t, ObjType>(size_t sz, ObjType ty)
        {
            auto pobj = std::make_shared<ObjArray>();
            pobj->data.size = sz;
            for (int i = 0; i < sz; i++)
                pobj->data.elems.push_back(create_obj_type(ty));
            pobj->init = true;
            return pobj;
        }

        template<>
        std::shared_ptr<ObjTuple> create_obj<ObjType::TUPLE>()
        {
            auto pobj = std::make_shared<ObjTuple>();
            pobj->init = false;
            return pobj;
        }

        template<>
        std::shared_ptr<ObjTuple> create_obj<ObjType::TUPLE, const std::vector<ObjType>&>(const std::vector<ObjType>& elems)
        {
            auto pobj = std::make_shared<ObjTuple>();
            for (auto ty : elems)
                pobj->data.elems.push_back(create_obj_type(ty));
            pobj->init = true;
            return pobj;
        }

        template<>
        std::shared_ptr<ObjTensor> create_obj<ObjType::TENSOR>()
        {

        }

        template<>
        std::shared_ptr<ObjDef> create_obj<ObjType::DEF>()
        {

        }

        template<>
        std::shared_ptr<ObjFn> create_obj<ObjType::FN>()
        {

        }

        template<>
        std::shared_ptr<ObjIntr> create_obj<ObjType::INTR>()
        {

        }
    }
}
