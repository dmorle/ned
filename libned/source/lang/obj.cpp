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
        bool ProgramHeap::create_type_bool(Errors& errs, Obj& obj)
        {
            return true;
        }

        bool ProgramHeap::create_type_fwidth(Errors& errs, Obj& obj)
        {
            return true;
        }

        bool ProgramHeap::create_type_int(Errors& errs, Obj& obj)
        {
            return true;
        }

        bool ProgramHeap::create_type_float(Errors& errs, Obj& obj)
        {
            return true;
        }

        bool ProgramHeap::create_type_str(Errors& errs, Obj& obj)
        {
            return true;
        }

        bool ProgramHeap::create_type_arr(Errors& errs, Obj& obj, TypeObj* ty)
        {
            return true;
        }

        bool ProgramHeap::create_type_agg(Errors& errs, Obj& obj, std::vector<TypeObj*> tys)
        {
            return true;
        }

        bool ProgramHeap::create_obj_bool(Errors& errs, Obj& obj, BoolObj val)
        {
            return true;
        }

        bool ProgramHeap::create_obj_fwidth(Errors& errs, Obj& obj, FWidthObj val)
        {
            return true;
        }

        bool ProgramHeap::create_obj_int(Errors& errs, Obj& obj, IntObj val)
        {
            return true;
        }

        bool ProgramHeap::create_obj_float(Errors& errs, Obj& obj, FloatObj val)
        {
            return true;
        }

        bool ProgramHeap::create_obj_str(Errors& errs, Obj& obj, const StrObj& val)
        {
            return true;
        }

        bool ProgramHeap::create_obj_agg(Errors& errs, Obj& obj, const AggObj& val)
        {
            return true;
        }

        bool ProgramHeap::create_obj_tensor(Errors& errs, Obj& obj, FWidthObj dty, const std::vector<IntObj>& dims)
        {
            return true;
        }

        bool TypeObj::cpy(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return true;
        }

        bool TypeObj::inst(Errors& errs, ProgramHeap& heap, Obj& dst)
        {
            return true;
        }

        bool TypeObj::set(Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::iadd(Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::isub(Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::imul(Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::idiv(Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::imod(Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::add(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::sub(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::mul(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::div(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::mod(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::eq(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::ne(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::ge(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::le(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::gt(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::lt(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::land(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::lor(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::idx(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::xstr(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return true;
        }

        bool TypeObj::xflt(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return true;
        }

        bool TypeObj::xint(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return true;
        }

        bool BoolType::cpy(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return heap.create_obj_bool(errs, dst, *src.bool_obj);
        }

        bool BoolType::inst(Errors& errs, ProgramHeap& heap, Obj& dst)
        {
            return heap.create_obj_bool(errs, dst, false);
        }

        bool BoolType::set(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            *dst.bool_obj = *src.bool_obj;
            return false;
        }

        bool BoolType::eq(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.bool_obj == *rhs.bool_obj);
        }

        bool BoolType::ne(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.bool_obj != *rhs.bool_obj);
        }

        bool BoolType::land(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.bool_obj && *rhs.bool_obj);
        }

        bool BoolType::lor(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.bool_obj || *rhs.bool_obj);
        }

        bool BoolType::xstr(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return heap.create_obj_str(errs, dst, *src.bool_obj ? "true" : "false");
        }

        bool FWidthType::cpy(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return heap.create_obj_fwidth(errs, dst, *src.fwidth_obj);
        }

        bool FWidthType::inst(Errors& errs, ProgramHeap& heap, Obj& dst)
        {
            return heap.create_obj_fwidth(errs, dst, core::tensor_dty::F32);
        }

        bool FWidthType::set(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            *dst.fwidth_obj = *src.fwidth_obj;
            return false;
        }

        bool FWidthType::eq(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.fwidth_obj == *rhs.fwidth_obj);
        }

        bool FWidthType::ne(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.fwidth_obj != *rhs.fwidth_obj);
        }

        bool FWidthType::xstr(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            std::string str;
            return
                core::dtype_str(*src.fwidth_obj, str) ||
                heap.create_obj_str(errs, dst, str);
        }

        bool IntType::cpy(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return heap.create_obj_int(errs, dst, *src.int_obj);
        }

        bool IntType::inst(Errors& errs, ProgramHeap& heap, Obj& dst)
        {
            return heap.create_obj_int(errs, dst, 0);
        }

        bool IntType::set(Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.int_obj = *rhs.int_obj;
            return false;
        }

        bool IntType::iadd(Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.int_obj += *rhs.int_obj;
            return false;
        }

        bool IntType::isub(Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.int_obj -= *rhs.int_obj;
            return false;
        }

        bool IntType::imul(Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.int_obj *= *rhs.int_obj;
            return false;
        }

        bool IntType::idiv(Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.int_obj /= *rhs.int_obj;
            return false;
        }

        bool IntType::imod(Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.int_obj %= *rhs.int_obj;
            return false;
        }

        bool IntType::add(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_int(errs, dst, *lhs.int_obj + *rhs.int_obj);
        }

        bool IntType::sub(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_int(errs, dst, *lhs.int_obj - *rhs.int_obj);
        }

        bool IntType::mul(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_int(errs, dst, *lhs.int_obj * *rhs.int_obj);
        }

        bool IntType::div(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_int(errs, dst, *lhs.int_obj / *rhs.int_obj);
        }

        bool IntType::mod(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_int(errs, dst, *lhs.int_obj % *rhs.int_obj);
        }

        bool IntType::eq(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.int_obj == *rhs.int_obj);
        }

        bool IntType::ne(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.int_obj != *rhs.int_obj);
        }

        bool IntType::ge(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.int_obj >= *rhs.int_obj);
        }

        bool IntType::le(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.int_obj <= *rhs.int_obj);
        }

        bool IntType::gt(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.int_obj > *rhs.int_obj);
        }

        bool IntType::lt(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.int_obj < *rhs.int_obj);
        }

        bool IntType::xstr(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            // TODO: implement int to string conversion
            return heap.create_obj_str(errs, dst, std::to_string(*src.int_obj));
        }

        bool IntType::xflt(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return heap.create_obj_float(errs, dst, *src.int_obj);
        }

        bool IntType::xint(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return heap.create_obj_int(errs, dst, *src.int_obj);
        }
        
        bool FloatType::cpy(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return heap.create_obj_float(errs, dst, *src.float_obj);
        }

        bool FloatType::inst(Errors& errs, ProgramHeap& heap, Obj& dst)
        {
            return heap.create_obj_float(errs, dst, 1.0);
        }

        bool FloatType::set(Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.float_obj = *rhs.float_obj;
            return false;
        }

        bool FloatType::iadd(Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.float_obj += *rhs.float_obj;
            return false;
        }

        bool FloatType::isub(Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.float_obj -= *rhs.float_obj;
            return false;
        }

        bool FloatType::imul(Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.float_obj *= *rhs.float_obj;
            return false;
        }

        bool FloatType::idiv(Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.float_obj /= *rhs.float_obj;
            return false;
        }

        bool FloatType::add(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_float(errs, dst, *lhs.float_obj + *rhs.float_obj);
        }

        bool FloatType::sub(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_float(errs, dst, *lhs.float_obj - *rhs.float_obj);
        }

        bool FloatType::mul(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_float(errs, dst, *lhs.float_obj * *rhs.float_obj);
        }

        bool FloatType::div(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_float(errs, dst, *lhs.float_obj / *rhs.float_obj);
        }

        bool FloatType::eq(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.float_obj == *rhs.float_obj);
        }

        bool FloatType::ne(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.float_obj != *rhs.float_obj);
        }

        bool FloatType::ge(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.float_obj >= *rhs.float_obj);
        }

        bool FloatType::le(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.float_obj <= *rhs.float_obj);
        }

        bool FloatType::gt(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.float_obj > *rhs.float_obj);
        }

        bool FloatType::lt(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.float_obj < *rhs.float_obj);
        }

        bool FloatType::xstr(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            // TODO: implement float to string conversion
            return heap.create_obj_str(errs, dst, std::to_string(*src.float_obj));
        }

        bool FloatType::xflt(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return heap.create_obj_float(errs, dst, *src.float_obj);
        }

        bool FloatType::xint(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return heap.create_obj_int(errs, dst, *src.float_obj);
        }

        bool StrType::cpy(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return heap.create_obj_str(errs, dst, *src.str_obj);
        }

        bool StrType::inst(Errors& errs, ProgramHeap& heap, Obj& dst)
        {
            return heap.create_obj_str(errs, dst, "");
        }

        bool StrType::set(Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.str_obj = *rhs.str_obj;
            return false;
        }

        bool StrType::iadd(Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.str_obj += *rhs.str_obj;
            return false;
        }

        bool StrType::add(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_str(errs, dst, *lhs.str_obj + *rhs.str_obj);
        }

        bool StrType::eq(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.str_obj == *rhs.str_obj);
        }

        bool StrType::ne(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.str_obj != *rhs.str_obj);
        }

        bool StrType::ge(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.str_obj >= *rhs.str_obj);
        }

        bool StrType::le(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.str_obj <= *rhs.str_obj);
        }

        bool StrType::gt(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.str_obj > *rhs.str_obj);
        }

        bool StrType::lt(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(errs, dst, *lhs.str_obj < *rhs.str_obj);
        }

        bool StrType::idx(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_str(errs, dst, std::string(1, (*lhs.str_obj)[*rhs.int_obj]));
        }

        bool StrType::xstr(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return heap.create_obj_str(errs, dst, *src.str_obj);
        }

        bool StrType::xflt(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            // TODO: implement string to float conversion
            return heap.create_obj_float(errs, dst, std::stod(*src.str_obj));
        }

        bool StrType::xint(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
        {
            // TODO: implement string to int conversion
            return heap.create_obj_int(errs, dst, std::stoll(*src.str_obj));
        }
        
        bool ArrType::cpy(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
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

        bool ArrType::inst(Errors& errs, ProgramHeap& heap, Obj& dst)
        {
            return heap.create_obj_agg(errs, dst, {});
        }

        bool ArrType::set(Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.agg_obj = *rhs.agg_obj;  // Copies the vector of pointers, not the data in the pointers
            return false;
        }

        bool ArrType::iadd(Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            lhs.agg_obj->insert(lhs.agg_obj->end(), rhs.agg_obj->begin(), rhs.agg_obj->end());
            return false;
        }

        bool ArrType::add(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            std::vector<Obj> objs;
            objs.insert(objs.end(), lhs.agg_obj->begin(), lhs.agg_obj->end());
            objs.insert(objs.end(), rhs.agg_obj->begin(), rhs.agg_obj->end());
            return heap.create_obj_agg(errs, dst, objs);
        }

        bool ArrType::eq(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
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

        bool ArrType::ne(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
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

        bool ArrType::idx(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            // TODO: bounds checking
            dst = lhs.agg_obj->operator[](*rhs.int_obj);
            return false;
        }

        bool ArrType::xstr(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
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

        bool AggType::cpy(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
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

        bool AggType::inst(Errors& errs, ProgramHeap& heap, Obj& dst)
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

        bool AggType::set(Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.agg_obj = *rhs.agg_obj;  // Copies the vector of pointers, not the data in the pointers
            return false;
        }

        bool AggType::eq(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
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

        bool AggType::ne(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
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

        bool AggType::idx(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            // This should be an error, not an assert
            assert(lhs.agg_obj->size() == elem_tys.size());

            // TODO: bounds checking
            dst = lhs.agg_obj->operator[](*rhs.int_obj);
            return false;
        }

        bool AggType::xstr(Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src)
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
