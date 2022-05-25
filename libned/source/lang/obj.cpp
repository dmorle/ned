#include <ned/errors.h>
#include <ned/lang/obj.h>
#include <ned/lang/interp.h>
#include <ned/lang/bytecode.h>
#include <ned/core/graph.h>

#include <string>
#include <cassert>
#include <unordered_set>
#include <functional>

namespace nn
{
    namespace lang
    {
        ProgramHeap::~ProgramHeap()
        {
            for (auto* ty : bool_types)
                delete ty;
            for (auto* ty : fty_types)
                delete ty;
            for (auto* ty : int_types)
                delete ty;
            for (auto* ty : float_types)
                delete ty;
            for (auto* ty : str_types)
                delete ty;
            for (auto* ty : arr_types)
                delete ty;
            for (auto* ty : agg_types)
                delete ty;

            for (auto* obj : bool_objs)
                delete obj;
            for (auto* obj : fty_objs)
                delete obj;
            for (auto* obj : int_objs)
                delete obj;
            for (auto* obj : float_objs)
                delete obj;
            for (auto* obj : str_objs)
                delete obj;
            for (auto* obj : agg_objs)
                delete obj;
        }

        bool ProgramHeap::create_type_bool(Obj& obj)
        {
            bool_types.push_back(new BoolType());
            obj.type_obj = bool_types.back();
            return false;
        }

        bool ProgramHeap::create_type_fty(Obj& obj)
        {
            fty_types.push_back(new FtyType());
            obj.type_obj = fty_types.back();
            return false;
        }

        bool ProgramHeap::create_type_int(Obj& obj)
        {
            int_types.push_back(new IntType());
            obj.type_obj = int_types.back();
            return false;
        }

        bool ProgramHeap::create_type_float(Obj& obj)
        {
            float_types.push_back(new FloatType());
            obj.type_obj = float_types.back();
            return false;
        }

        bool ProgramHeap::create_type_str(Obj& obj)
        {
            str_types.push_back(new StrType());
            obj.type_obj = str_types.back();
            return false;
        }

        bool ProgramHeap::create_type_arr(Obj& obj, TypeObj* ty)
        {
            arr_types.push_back(new ArrType());
            arr_types.back()->elem_ty = ty;
            obj.type_obj = arr_types.back();
            return false;
        }

        bool ProgramHeap::create_type_agg(Obj& obj, std::vector<TypeObj*> tys)
        {
            agg_types.push_back(new AggType());
            agg_types.back()->elem_tys = std::move(tys);
            obj.type_obj = agg_types.back();
            return false;
        }

        bool ProgramHeap::create_obj_bool(Obj& obj, BoolObj val)
        {
            bool_objs.push_back(new BoolObj(val));
            obj.bool_obj = bool_objs.back();
            return false;
        }

        bool ProgramHeap::create_obj_fty(Obj& obj, FtyObj val)
        {
            fty_objs.push_back(new FtyObj(val));
            obj.fty_obj = fty_objs.back();
            return false;
        }

        bool ProgramHeap::create_obj_int(Obj& obj, IntObj val)
        {
            int_objs.push_back(new IntObj(val));
            obj.int_obj = int_objs.back();
            return false;
        }

        bool ProgramHeap::create_obj_float(Obj& obj, FloatObj val)
        {
            float_objs.push_back(new FloatObj(val));
            obj.float_obj = float_objs.back();
            return false;
        }

        bool ProgramHeap::create_obj_str(Obj& obj, const StrObj& val)
        {
            str_objs.push_back(new StrObj(val));
            obj.str_obj = str_objs.back();
            return false;
        }

        bool ProgramHeap::create_obj_agg(Obj& obj, const AggObj& val)
        {
            agg_objs.push_back(new AggObj(val));
            obj.agg_obj = agg_objs.back();
            return false;
        }

        bool TypeObj::cpy(ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return true;
        }

        bool TypeObj::inst(ProgramHeap& heap, Obj& dst)
        {
            return true;
        }

        bool TypeObj::set(ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::iadd(ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::isub(ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::imul(ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::idiv(ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::imod(ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::add(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::sub(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::mul(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::div(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::mod(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::eq(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::ne(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::ge(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::le(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::gt(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::lt(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::land(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::lor(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::idx(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return true;
        }

        bool TypeObj::len(ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return true;
        }

        bool TypeObj::xstr(ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return true;
        }

        bool TypeObj::xflt(ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return true;
        }

        bool TypeObj::xint(ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return true;
        }

        bool BoolType::cpy(ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return heap.create_obj_bool(dst, *src.bool_obj);
        }

        bool BoolType::inst(ProgramHeap& heap, Obj& dst)
        {
            return heap.create_obj_bool(dst, false);
        }

        bool BoolType::set(ProgramHeap& heap, Obj& dst, const Obj src)
        {
            *dst.bool_obj = *src.bool_obj;
            return false;
        }

        bool BoolType::eq(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(dst, *lhs.bool_obj == *rhs.bool_obj);
        }

        bool BoolType::ne(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(dst, *lhs.bool_obj != *rhs.bool_obj);
        }

        bool BoolType::land(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(dst, *lhs.bool_obj && *rhs.bool_obj);
        }

        bool BoolType::lor(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(dst, *lhs.bool_obj || *rhs.bool_obj);
        }

        bool BoolType::xstr(ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return heap.create_obj_str(dst, *src.bool_obj ? "true" : "false");
        }

        bool BoolType::cfg(core::Config*& cfg, const Obj src)
        {
            cfg = new core::BoolConfig(*src.bool_obj);
            return false;
        }

        bool FtyType::cpy(ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return heap.create_obj_fty(dst, *src.fty_obj);
        }

        bool FtyType::inst(ProgramHeap& heap, Obj& dst)
        {
            return heap.create_obj_fty(dst, core::EdgeFty::F32);
        }

        bool FtyType::set(ProgramHeap& heap, Obj& dst, const Obj src)
        {
            *dst.fty_obj = *src.fty_obj;
            return false;
        }

        bool FtyType::eq(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(dst, *lhs.fty_obj == *rhs.fty_obj);
        }

        bool FtyType::ne(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(dst, *lhs.fty_obj != *rhs.fty_obj);
        }

        bool FtyType::xstr(ProgramHeap& heap, Obj& dst, const Obj src)
        {
            std::string str;
            return
                core::fty_str(*src.fty_obj, str) ||
                heap.create_obj_str(dst, str);
        }

        bool FtyType::cfg(core::Config*& cfg, const Obj src)
        {
            cfg = new core::FtyConfig(*src.fty_obj);
            return false;
        }

        bool IntType::cpy(ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return heap.create_obj_int(dst, *src.int_obj);
        }

        bool IntType::inst(ProgramHeap& heap, Obj& dst)
        {
            return heap.create_obj_int(dst, 0);
        }

        bool IntType::set(ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.int_obj = *rhs.int_obj;
            return false;
        }

        bool IntType::iadd(ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.int_obj += *rhs.int_obj;
            return false;
        }

        bool IntType::isub(ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.int_obj -= *rhs.int_obj;
            return false;
        }

        bool IntType::imul(ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.int_obj *= *rhs.int_obj;
            return false;
        }

        bool IntType::idiv(ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.int_obj /= *rhs.int_obj;
            return false;
        }

        bool IntType::imod(ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.int_obj %= *rhs.int_obj;
            return false;
        }

        bool IntType::add(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_int(dst, *lhs.int_obj + *rhs.int_obj);
        }

        bool IntType::sub(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_int(dst, *lhs.int_obj - *rhs.int_obj);
        }

        bool IntType::mul(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_int(dst, *lhs.int_obj * *rhs.int_obj);
        }

        bool IntType::div(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_int(dst, *lhs.int_obj / *rhs.int_obj);
        }

        bool IntType::mod(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_int(dst, *lhs.int_obj % *rhs.int_obj);
        }

        bool IntType::eq(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(dst, *lhs.int_obj == *rhs.int_obj);
        }

        bool IntType::ne(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(dst, *lhs.int_obj != *rhs.int_obj);
        }

        bool IntType::ge(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(dst, *lhs.int_obj >= *rhs.int_obj);
        }

        bool IntType::le(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(dst, *lhs.int_obj <= *rhs.int_obj);
        }

        bool IntType::gt(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(dst, *lhs.int_obj > *rhs.int_obj);
        }

        bool IntType::lt(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(dst, *lhs.int_obj < *rhs.int_obj);
        }

        bool IntType::xstr(ProgramHeap& heap, Obj& dst, const Obj src)
        {
            // TODO: implement int to string conversion
            return heap.create_obj_str(dst, std::to_string(*src.int_obj));
        }

        bool IntType::xflt(ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return heap.create_obj_float(dst, (FloatObj)*src.int_obj);
        }

        bool IntType::xint(ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return heap.create_obj_int(dst, *src.int_obj);
        }

        bool IntType::cfg(core::Config*& cfg, const Obj src)
        {
            cfg = new core::IntConfig(*src.int_obj);
            return false;
        }
        
        bool FloatType::cpy(ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return heap.create_obj_float(dst, *src.float_obj);
        }

        bool FloatType::inst(ProgramHeap& heap, Obj& dst)
        {
            return heap.create_obj_float(dst, 1.0);
        }

        bool FloatType::set(ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.float_obj = *rhs.float_obj;
            return false;
        }

        bool FloatType::iadd(ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.float_obj += *rhs.float_obj;
            return false;
        }

        bool FloatType::isub(ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.float_obj -= *rhs.float_obj;
            return false;
        }

        bool FloatType::imul(ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.float_obj *= *rhs.float_obj;
            return false;
        }

        bool FloatType::idiv(ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.float_obj /= *rhs.float_obj;
            return false;
        }

        bool FloatType::add(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_float(dst, *lhs.float_obj + *rhs.float_obj);
        }

        bool FloatType::sub(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_float(dst, *lhs.float_obj - *rhs.float_obj);
        }

        bool FloatType::mul(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_float(dst, *lhs.float_obj * *rhs.float_obj);
        }

        bool FloatType::div(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_float(dst, *lhs.float_obj / *rhs.float_obj);
        }

        bool FloatType::eq(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(dst, *lhs.float_obj == *rhs.float_obj);
        }

        bool FloatType::ne(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(dst, *lhs.float_obj != *rhs.float_obj);
        }

        bool FloatType::ge(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(dst, *lhs.float_obj >= *rhs.float_obj);
        }

        bool FloatType::le(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(dst, *lhs.float_obj <= *rhs.float_obj);
        }

        bool FloatType::gt(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(dst, *lhs.float_obj > *rhs.float_obj);
        }

        bool FloatType::lt(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(dst, *lhs.float_obj < *rhs.float_obj);
        }

        bool FloatType::xstr(ProgramHeap& heap, Obj& dst, const Obj src)
        {
            // TODO: implement float to string conversion
            return heap.create_obj_str(dst, std::to_string(*src.float_obj));
        }

        bool FloatType::xflt(ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return heap.create_obj_float(dst, *src.float_obj);
        }

        bool FloatType::xint(ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return heap.create_obj_int(dst, (IntObj)*src.float_obj);
        }

        bool FloatType::cfg(core::Config*& cfg, const Obj src)
        {
            cfg = new core::FloatConfig(*src.float_obj);
            return false;
        }

        bool StrType::cpy(ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return heap.create_obj_str(dst, *src.str_obj);
        }

        bool StrType::inst(ProgramHeap& heap, Obj& dst)
        {
            return heap.create_obj_str(dst, "");
        }

        bool StrType::set(ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.str_obj = *rhs.str_obj;
            return false;
        }

        bool StrType::iadd(ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.str_obj += *rhs.str_obj;
            return false;
        }

        bool StrType::add(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_str(dst, *lhs.str_obj + *rhs.str_obj);
        }

        bool StrType::eq(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(dst, *lhs.str_obj == *rhs.str_obj);
        }

        bool StrType::ne(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(dst, *lhs.str_obj != *rhs.str_obj);
        }

        bool StrType::ge(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(dst, *lhs.str_obj >= *rhs.str_obj);
        }

        bool StrType::le(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(dst, *lhs.str_obj <= *rhs.str_obj);
        }

        bool StrType::gt(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(dst, *lhs.str_obj > *rhs.str_obj);
        }

        bool StrType::lt(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_bool(dst, *lhs.str_obj < *rhs.str_obj);
        }

        bool StrType::idx(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            return heap.create_obj_str(dst, std::string(1, (*lhs.str_obj)[*rhs.int_obj]));
        }

        bool StrType::xstr(ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return heap.create_obj_str(dst, *src.str_obj);
        }

        bool StrType::xflt(ProgramHeap& heap, Obj& dst, const Obj src)
        {
            // TODO: implement string to float conversion
            return heap.create_obj_float(dst, std::stod(*src.str_obj));
        }

        bool StrType::xint(ProgramHeap& heap, Obj& dst, const Obj src)
        {
            // TODO: implement string to int conversion
            return heap.create_obj_int(dst, std::stoll(*src.str_obj));
        }
        
        bool StrType::cfg(core::Config*& cfg, const Obj src)
        {
            cfg = new core::StringConfig(*src.str_obj);
            return false;
        }

        bool ArrType::cpy(ProgramHeap& heap, Obj& dst, const Obj src)
        {
            std::vector<Obj> objs;
            for (const Obj e_src : *src.agg_obj)
            {
                Obj e_dst;
                if (elem_ty->cpy(heap, e_dst, e_src))
                    return true;
                objs.push_back(e_dst);
            }
            return heap.create_obj_agg(dst, objs);
        }

        bool ArrType::inst(ProgramHeap& heap, Obj& dst)
        {
            return heap.create_obj_agg(dst, {});
        }

        bool ArrType::set(ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.agg_obj = *rhs.agg_obj;  // Copies the vector of pointers, not the data in the pointers
            return false;
        }

        bool ArrType::iadd(ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            lhs.agg_obj->insert(lhs.agg_obj->end(), rhs.agg_obj->begin(), rhs.agg_obj->end());
            return false;
        }

        bool ArrType::add(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            std::vector<Obj> objs;
            objs.insert(objs.end(), lhs.agg_obj->begin(), lhs.agg_obj->end());
            objs.insert(objs.end(), rhs.agg_obj->begin(), rhs.agg_obj->end());
            return heap.create_obj_agg(dst, objs);
        }

        bool ArrType::eq(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            if (lhs.agg_obj->size() != rhs.agg_obj->size())
                return heap.create_obj_bool(dst, false);

            for (size_t i = 0; i < lhs.agg_obj->size(); i++)
            {
                Obj e;
                if (elem_ty->eq(heap, e, lhs.agg_obj->operator[](i), rhs.agg_obj->operator[](i)))
                    return true;
                if (!*e.bool_obj)
                    return heap.create_obj_bool(dst, false);
            }
            return heap.create_obj_bool(dst, true);
        }

        bool ArrType::ne(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            if (lhs.agg_obj->size() != rhs.agg_obj->size())
                return heap.create_obj_bool(dst, true);

            for (size_t i = 0; i < lhs.agg_obj->size(); i++)
            {
                Obj e;
                if (elem_ty->eq(heap, e, lhs.agg_obj->operator[](i), rhs.agg_obj->operator[](i)))
                    return true;
                if (!*e.bool_obj)
                    return heap.create_obj_bool(dst, true);
            }
            return heap.create_obj_bool(dst, false);
        }

        bool ArrType::idx(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            IntObj idx = *rhs.int_obj;
            if (idx < 0)  // Allowing for indexing off the back of the array
                idx += lhs.agg_obj->size();

            if (idx < 0 || lhs.agg_obj->size() <= (size_t)idx)
                return error::runtime("Index out of bounds");
            dst = lhs.agg_obj->operator[](*rhs.int_obj);
            return false;
        }

        bool ArrType::len(ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return heap.create_obj_int(dst, src.agg_obj->size());
        }

        bool ArrType::xstr(ProgramHeap& heap, Obj& dst, const Obj src)
        {
            if (src.agg_obj->size() == 0)
                return heap.create_obj_str(dst, "[]");

            auto it = src.agg_obj->begin();
            Obj str;
            if (elem_ty->xstr(heap, str, *it))
                return true;

            std::stringstream ss;
            ss << '[' << *str.str_obj;
            for (it++; it != src.agg_obj->end(); it++)
            {
                if (elem_ty->xstr(heap, str, *it))
                    return true;
                ss << ", " << *str.str_obj;
            }
            ss << ']';
            return heap.create_obj_str(dst, ss.str());
        }

        bool ArrType::cfg(core::Config*& cfg, const Obj src)
        {
            std::vector<core::Config*> configs(src.agg_obj->size());
            for (Obj e : *src.agg_obj)
            {
                core::Config* elem_cfg;
                if (elem_ty->cfg(elem_cfg, e))
                {
                    for (core::Config* cfg : configs)
                        delete cfg;
                    return true;
                }
                configs.push_back(elem_cfg);
            }
            cfg = new core::ListConfig(configs);
            return false;
        }

        bool AggType::cpy(ProgramHeap& heap, Obj& dst, const Obj src)
        {
            // This should be an error, not an assert
            assert(elem_tys.size() == 0 || src.agg_obj->size() == elem_tys.size());
            std::vector<Obj> objs;
            for (size_t i = 0; i < elem_tys.size(); i++)
            {
                Obj e;
                if (elem_tys[i]->cpy(heap, e, src.agg_obj->operator[](i)))
                    return true;
                objs.push_back(e);
            }
            return heap.create_obj_agg(dst, objs);
        }

        bool AggType::inst(ProgramHeap& heap, Obj& dst)
        {
            std::vector<Obj> objs;
            for (TypeObj* ty : elem_tys)
            {
                Obj e_dst;
                if (ty->inst(heap, e_dst))
                    return true;
                objs.push_back(e_dst);
            }
            return heap.create_obj_agg(dst, objs);
        }

        bool AggType::set(ProgramHeap& heap, Obj& lhs, const Obj rhs)
        {
            *lhs.agg_obj = *rhs.agg_obj;  // Copies the vector of pointers, not the data in the pointers
            return false;
        }

        bool AggType::eq(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            // These should be RuntimeErrors, not asserts
            assert(lhs.agg_obj->size() == elem_tys.size());
            assert(rhs.agg_obj->size() == elem_tys.size());

            for (size_t i = 0; i < elem_tys.size(); i++)
            {
                Obj e;
                if (elem_tys[i]->eq(heap, e, lhs.agg_obj->operator[](i), rhs.agg_obj->operator[](i)))
                    return true;
                if (!*e.bool_obj)
                    return heap.create_obj_bool(dst, false);
            }
            return heap.create_obj_bool(dst, true);
        }

        bool AggType::ne(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            // This should be an error, not an assert
            assert(lhs.agg_obj->size() == elem_tys.size());
            assert(rhs.agg_obj->size() == elem_tys.size());

            for (size_t i = 0; i < elem_tys.size(); i++)
            {
                Obj e;
                if (elem_tys[i]->eq(heap, e, lhs.agg_obj->operator[](i), rhs.agg_obj->operator[](i)))
                    return true;
                if (!*e.bool_obj)
                    return heap.create_obj_bool(dst, true);
            }
            return heap.create_obj_bool(dst, false);
        }

        bool AggType::idx(ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs)
        {
            // Tuples are always initialized, and bounds are checked for all aggregate types at compile time
            // This can only occur in compiled bytecode when a struct that wasn't initialized was dereferenced
            if (elem_tys.size() == 0)
                return error::runtime("Dereferencing uninitialized struct");

            // This should never occur from compiled bytecode, only manually written bytecode
            assert(elem_tys.size() == lhs.agg_obj->size());

            dst = lhs.agg_obj->operator[](*rhs.int_obj);
            return false;
        }

        bool ArrType::len(ProgramHeap& heap, Obj& dst, const Obj src)
        {
            return heap.create_obj_int(dst, src.agg_obj->size());
        }

        bool AggType::xstr(ProgramHeap& heap, Obj& dst, const Obj src)
        {
            if (src.agg_obj->size() == 0)  // Uninitialized struct members of structs
                return heap.create_obj_str(dst, "()");

            if (src.agg_obj->size() != elem_tys.size())
                return error::runtime("type object mismatch in aggregate object");

            Obj str;
            if (elem_tys[0]->xstr(heap, str, src.agg_obj->operator[](0)))
                return true;

            std::stringstream ss;
            ss << '(' << *str.str_obj;
            for (size_t i = 1; i < elem_tys.size(); i++)
            {
                if (elem_tys[i]->xstr(heap, str, src.agg_obj->operator[](i)))
                    return true;
                ss << ", " << *str.str_obj;
            }
            ss << ')';
            return heap.create_obj_str(dst, ss.str());
        }

        bool AggType::cfg(core::Config*& cfg, const Obj src)
        {
            if (src.agg_obj->size() != elem_tys.size())
            {
                if (src.agg_obj->size() == 0)  // Probably an uninitialized struct
                    return error::runtime("type object mismatch in aggregate object - likely an uninitialized compound struct");
                return error::runtime("type object mismatch in aggregate object");
            }

            std::vector<core::Config*> configs(src.agg_obj->size());
            for (size_t i = 0; i < src.agg_obj->size(); i++)
            {
                core::Config* elem_cfg;
                if (elem_tys[i]->cfg(elem_cfg, src.agg_obj->operator[](i)))
                {
                    for (core::Config* cfg : configs)
                        delete cfg;
                    return true;
                }
                configs.push_back(elem_cfg);
            }
            cfg = new core::ListConfig(configs);
            return false;
        }
    }
}
