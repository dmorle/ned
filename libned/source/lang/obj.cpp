#include <ned/lang/obj.h>
#include <ned/lang/interp.h>
#include <ned/core/tensor.h>
#include <ned/core/graph.h>

#include <string>
#include <cassert>

namespace nn
{
    namespace lang
    {
        bool ProgramHeap::create_type_bool(RuntimeErrors& errs, Obj& obj)
        {
            return true;
        }

        bool ProgramHeap::create_type_fwidth(RuntimeErrors& errs, Obj& obj)
        {
            return true;
        }

        bool ProgramHeap::create_type_int(RuntimeErrors& errs, Obj& obj)
        {
            return true;
        }

        bool ProgramHeap::create_type_float(RuntimeErrors& errs, Obj& obj)
        {
            return true;
        }

        bool ProgramHeap::create_type_str(RuntimeErrors& errs, Obj& obj)
        {
            return true;
        }

        bool ProgramHeap::create_type_arr(RuntimeErrors& errs, Obj& obj, TypeObj* ty)
        {
            return true;
        }

        bool ProgramHeap::create_type_agg(RuntimeErrors& errs, Obj& obj, std::vector<TypeObj*> tys)
        {
            return true;
        }

        bool ProgramHeap::create_obj_bool(RuntimeErrors& errs, Obj& obj, BoolObj val)
        {
            return true;
        }

        bool ProgramHeap::create_obj_fwidth(RuntimeErrors& errs, Obj& obj, FWidthObj val)
        {
            return true;
        }

        bool ProgramHeap::create_obj_int(RuntimeErrors& errs, Obj& obj, IntObj val)
        {
            return true;
        }

        bool ProgramHeap::create_obj_float(RuntimeErrors& errs, Obj& obj, FloatObj val)
        {
            return true;
        }

        bool ProgramHeap::create_obj_str(RuntimeErrors& errs, Obj& obj, const StrObj& val)
        {
            return true;
        }

        bool ProgramHeap::create_obj_agg(RuntimeErrors& errs, Obj& obj, const AggObj& val)
        {
            return true;
        }

        bool ProgramHeap::create_obj_tensor(RuntimeErrors& errs, Obj& obj, FWidthObj dty, const std::vector<IntObj>& dims)
        {
            return true;
        }

        bool TypeObj::cpy(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return true;
        }

        bool TypeObj::inst(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst)
        {
            return true;
        }

        bool TypeObj::set(RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::iadd(RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::isub(RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::imul(RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::idiv(RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::imod(RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::add(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::sub(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::mul(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::div(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::mod(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::eq(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::ne(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::ge(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::le(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::gt(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::lt(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::land(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::lor(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::idx(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::xstr(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return true;
        }

        bool TypeObj::xflt(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return true;
        }

        bool TypeObj::xint(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return true;
        }

        bool BoolType::cpy(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return heap.create_obj_bool(errs, dst, *src.bool_obj);
        }

        bool BoolType::inst(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst)
        {
            return heap.create_obj_bool(errs, dst, false);
        }

        bool BoolType::set(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            *dst.bool_obj = *src.bool_obj;
            return false;
        }

        bool BoolType::eq(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.bool_obj == *rhs.bool_obj);
        }

        bool BoolType::ne(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.bool_obj != *rhs.bool_obj);
        }

        bool BoolType::land(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.bool_obj && *rhs.bool_obj);
        }

        bool BoolType::lor(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.bool_obj || *rhs.bool_obj);
        }

        bool BoolType::xstr(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return heap.create_obj_str(errs, dst, *src.bool_obj ? "true" : "false");
        }

        bool FWidthType::cpy(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return heap.create_obj_fwidth(errs, dst, *src.fwidth_obj);
        }

        bool FWidthType::inst(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst)
        {
            return heap.create_obj_fwidth(errs, dst, core::tensor_dty::F32);
        }

        bool FWidthType::set(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            *dst.fwidth_obj = *src.fwidth_obj;
            return false;
        }

        bool FWidthType::eq(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.fwidth_obj == *rhs.fwidth_obj);
        }

        bool FWidthType::ne(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.fwidth_obj != *rhs.fwidth_obj);
        }

        bool FWidthType::xstr(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            std::string str;
            return
                core::dtype_str(*src.fwidth_obj, str) ||
                heap.create_obj_str(errs, dst, str);
        }

        bool IntType::cpy(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return heap.create_obj_int(errs, dst, *src.int_obj);
        }

        bool IntType::inst(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst)
        {
            return heap.create_obj_int(errs, dst, 0);
        }

        bool IntType::set(RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.int_obj = *rhs.int_obj;
            return false;
        }

        bool IntType::iadd(RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.int_obj += *rhs.int_obj;
            return false;
        }

        bool IntType::isub(RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.int_obj -= *rhs.int_obj;
            return false;
        }

        bool IntType::imul(RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.int_obj *= *rhs.int_obj;
            return false;
        }

        bool IntType::idiv(RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.int_obj /= *rhs.int_obj;
            return false;
        }

        bool IntType::imod(RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.int_obj %= *rhs.int_obj;
            return false;
        }

        bool IntType::add(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_int(errs, dst, *lhs.int_obj + *rhs.int_obj);
        }

        bool IntType::sub(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_int(errs, dst, *lhs.int_obj - *rhs.int_obj);
        }

        bool IntType::mul(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_int(errs, dst, *lhs.int_obj * *rhs.int_obj);
        }

        bool IntType::div(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_int(errs, dst, *lhs.int_obj / *rhs.int_obj);
        }

        bool IntType::mod(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_int(errs, dst, *lhs.int_obj % *rhs.int_obj);
        }

        bool IntType::eq(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.int_obj == *rhs.int_obj);
        }

        bool IntType::ne(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.int_obj != *rhs.int_obj);
        }

        bool IntType::ge(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.int_obj >= *rhs.int_obj);
        }

        bool IntType::le(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.int_obj <= *rhs.int_obj);
        }

        bool IntType::gt(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.int_obj > *rhs.int_obj);
        }

        bool IntType::lt(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.int_obj < *rhs.int_obj);
        }

        bool IntType::xstr(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            // TODO: implement int to string conversion
            return heap.create_obj_str(errs, dst, std::to_string(*src.int_obj));
        }

        bool IntType::xflt(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return heap.create_obj_float(errs, dst, *src.int_obj);
        }

        bool IntType::xint(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return heap.create_obj_int(errs, dst, *src.int_obj);
        }
        
        bool FloatType::cpy(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return heap.create_obj_float(errs, dst, *src.float_obj);
        }

        bool FloatType::inst(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst)
        {
            return heap.create_obj_float(errs, dst, 1.0);
        }

        bool FloatType::set(RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.float_obj = *rhs.float_obj;
            return false;
        }

        bool FloatType::iadd(RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.float_obj += *rhs.float_obj;
            return false;
        }

        bool FloatType::isub(RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.float_obj -= *rhs.float_obj;
            return false;
        }

        bool FloatType::imul(RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.float_obj *= *rhs.float_obj;
            return false;
        }

        bool FloatType::idiv(RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.float_obj /= *rhs.float_obj;
            return false;
        }

        bool FloatType::add(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_float(errs, dst, *lhs.float_obj + *rhs.float_obj);
        }

        bool FloatType::sub(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_float(errs, dst, *lhs.float_obj - *rhs.float_obj);
        }

        bool FloatType::mul(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_float(errs, dst, *lhs.float_obj * *rhs.float_obj);
        }

        bool FloatType::div(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_float(errs, dst, *lhs.float_obj / *rhs.float_obj);
        }

        bool FloatType::eq(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.float_obj == *rhs.float_obj);
        }

        bool FloatType::ne(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.float_obj != *rhs.float_obj);
        }

        bool FloatType::ge(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.float_obj >= *rhs.float_obj);
        }

        bool FloatType::le(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.float_obj <= *rhs.float_obj);
        }

        bool FloatType::gt(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.float_obj > *rhs.float_obj);
        }

        bool FloatType::lt(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.float_obj < *rhs.float_obj);
        }

        bool FloatType::xstr(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            // TODO: implement float to string conversion
            return heap.create_obj_str(errs, dst, std::to_string(*src.float_obj));
        }

        bool FloatType::xflt(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return heap.create_obj_float(errs, dst, *src.float_obj);
        }

        bool FloatType::xint(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return heap.create_obj_int(errs, dst, *src.float_obj);
        }

        bool StrType::cpy(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return heap.create_obj_str(errs, dst, *src.str_obj);
        }

        bool StrType::inst(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst)
        {
            return heap.create_obj_str(errs, dst, "");
        }

        bool StrType::set(RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.str_obj = *rhs.str_obj;
            return false;
        }

        bool StrType::iadd(RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.str_obj += *rhs.str_obj;
            return false;
        }

        bool StrType::add(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_str(errs, dst, *lhs.str_obj + *rhs.str_obj);
        }

        bool StrType::eq(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.str_obj == *rhs.str_obj);
        }

        bool StrType::ne(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.str_obj != *rhs.str_obj);
        }

        bool StrType::ge(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.str_obj >= *rhs.str_obj);
        }

        bool StrType::le(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.str_obj <= *rhs.str_obj);
        }

        bool StrType::gt(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.str_obj > *rhs.str_obj);
        }

        bool StrType::lt(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.str_obj < *rhs.str_obj);
        }

        bool StrType::idx(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_str(errs, dst, std::string(1, (*lhs.str_obj)[*rhs.int_obj]));
        }

        bool StrType::xstr(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return heap.create_obj_str(errs, dst, *src.str_obj);
        }

        bool StrType::xflt(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            // TODO: implement string to float conversion
            return heap.create_obj_float(errs, dst, std::stod(*src.str_obj));
        }

        bool StrType::xint(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            // TODO: implement string to int conversion
            return heap.create_obj_int(errs, dst, std::stoll(*src.str_obj));
        }
        
        bool ArrType::cpy(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            std::vector<Obj> objs;
            for (const Obj e_src : *src.agg_obj)
            {
                Obj e_dst;
                if (elem_ty->cpy(errs, heap, e_dst, e_src))
                    return true;
                objs.push_back(e_dst);
            }
            return heap.create_obj_agg(errs, dst, objs);
        }

        bool ArrType::inst(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst)
        {
            return heap.create_obj_agg(errs, dst, {});
        }

        bool ArrType::set(RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.agg_obj = *rhs.agg_obj;  // Copies the vector of pointers, not the data in the pointers
            return false;
        }

        bool ArrType::iadd(RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            lhs.agg_obj->insert(lhs.agg_obj->end(), rhs.agg_obj->begin(), rhs.agg_obj->end());
            return false;
        }

        bool ArrType::add(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            std::vector<Obj> objs;
            objs.insert(objs.end(), lhs.agg_obj->begin(), lhs.agg_obj->end());
            objs.insert(objs.end(), rhs.agg_obj->begin(), rhs.agg_obj->end());
            return heap.create_obj_agg(errs, dst, objs);
        }

        bool ArrType::eq(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            if (lhs.agg_obj->size() != rhs.agg_obj->size())
                return heap.create_obj_bool(errs, dst, false);

            for (size_t i = 0; i < lhs.agg_obj->size(); i++)
            {
                Obj e;
                if (elem_ty->eq(errs, heap, e, lhs.agg_obj->operator[](i), rhs.agg_obj->operator[](i)))
                    return true;
                if (!*e.bool_obj)
                    return heap.create_obj_bool(errs, dst, false);
            }
            return heap.create_obj_bool(errs, dst, true);
        }

        bool ArrType::ne(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            if (lhs.agg_obj->size() != rhs.agg_obj->size())
                return heap.create_obj_bool(errs, dst, true);

            for (size_t i = 0; i < lhs.agg_obj->size(); i++)
            {
                Obj e;
                if (elem_ty->eq(errs, heap, e, lhs.agg_obj->operator[](i), rhs.agg_obj->operator[](i)))
                    return true;
                if (!*e.bool_obj)
                    return heap.create_obj_bool(errs, dst, true);
            }
            return heap.create_obj_bool(errs, dst, false);
        }

        bool ArrType::idx(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            // TODO: bounds checking
            dst = lhs.agg_obj->operator[](*rhs.int_obj);
            return false;
        }

        bool ArrType::xstr(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            if (src.agg_obj->size() == 0)
                return heap.create_obj_str(errs, dst, "[]");

            auto it = src.agg_obj->begin();
            Obj str;
            if (elem_ty->xstr(errs, heap, str, *it))
                return true;

            std::stringstream ss;
            ss << '[' << *str.str_obj;
            for (it++; it != src.agg_obj->end(); it++)
            {
                if (elem_ty->xstr(errs, heap, str, *it))
                    return true;
                ss << ", " << *str.str_obj;
            }
            ss << ']';
            return heap.create_obj_str(errs, dst, ss.str());
        }

        bool AggType::cpy(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            // This should be an error, not an assert
            assert(elem_tys.size() == 0 || src.agg_obj->size() == elem_tys.size());
            std::vector<Obj> objs;
            for (size_t i = 0; i < elem_tys.size(); i++)
            {
                Obj e;
                if (elem_tys[i]->cpy(errs, heap, e, src.agg_obj->operator[](i)))
                    return true;
                objs.push_back(e);
            }
            return heap.create_obj_agg(errs, dst, objs);
        }

        bool AggType::inst(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst)
        {
            std::vector<Obj> objs;
            for (TypeObj* ty : elem_tys)
            {
                Obj e_dst;
                if (ty->inst(errs, heap, e_dst))
                    return true;
                objs.push_back(e_dst);
            }
            return heap.create_obj_agg(errs, dst, objs);
        }

        bool AggType::set(RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.agg_obj = *rhs.agg_obj;  // Copies the vector of pointers, not the data in the pointers
            return false;
        }

        bool AggType::eq(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            // These should be errors, not asserts
            assert(lhs.agg_obj->size() == elem_tys.size());
            assert(rhs.agg_obj->size() == elem_tys.size());

            for (size_t i = 0; i < elem_tys.size(); i++)
            {
                Obj e;
                if (elem_tys[i]->eq(errs, heap, e, lhs.agg_obj->operator[](i), rhs.agg_obj->operator[](i)))
                    return true;
                if (!*e.bool_obj)
                    return heap.create_obj_bool(errs, dst, false);
            }
            return heap.create_obj_bool(errs, dst, true);
        }

        bool AggType::ne(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            // This should be an error, not an assert
            assert(lhs.agg_obj->size() == elem_tys.size());
            assert(rhs.agg_obj->size() == elem_tys.size());

            for (size_t i = 0; i < elem_tys.size(); i++)
            {
                Obj e;
                if (elem_tys[i]->eq(errs, heap, e, lhs.agg_obj->operator[](i), rhs.agg_obj->operator[](i)))
                    return true;
                if (!*e.bool_obj)
                    return heap.create_obj_bool(errs, dst, true);
            }
            return heap.create_obj_bool(errs, dst, false);
        }

        bool AggType::idx(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            // This should be an error, not an assert
            assert(lhs.agg_obj->size() == elem_tys.size());

            // TODO: bounds checking
            dst = lhs.agg_obj->operator[](*rhs.int_obj);
            return false;
        }

        bool AggType::xstr(RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            // This should be an error, not an assert
            assert(src.agg_obj->size() == elem_tys.size());

            if (src.agg_obj->size() == 0)
                return heap.create_obj_str(errs, dst, "()");

            Obj str;
            if (elem_tys[0]->xstr(errs, heap, str, src.agg_obj->operator[](0)))
                return true;

            std::stringstream ss;
            ss << '(' << *str.str_obj;
            for (size_t i = 1; i < elem_tys.size(); i++)
            {
                if (elem_tys[i]->xstr(errs, heap, str, src.agg_obj->operator[](i)))
                    return true;
                ss << ", " << *str.str_obj;
            }
            ss << ')';
            return heap.create_obj_str(errs, dst, ss.str());
        }
    }
}
