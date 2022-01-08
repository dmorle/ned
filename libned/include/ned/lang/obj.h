#ifndef NED_OBJ_H
#define NED_OBJ_H

#include <ned/lang/ast.h>
#include <ned/core/graph.h>

#include <string>
#include <sstream>
#include <unordered_map>

namespace nn
{
    namespace lang
    {
        class TypeObj;
        using BoolObj = bool;
        using FWidthObj = core::tensor_dty;
        using IntObj = int64_t;
        using FloatObj = double;
        using StrObj = std::string;
        using AggObj = std::vector<Obj>;
        using TenObj = core::Edge;

        union Obj
        {
            TypeObj   *type_obj;
            BoolObj   *bool_obj;
            FWidthObj *fwidth_obj;
            IntObj    *int_obj;
            FloatObj  *float_obj;
            StrObj    *str_obj;
            AggObj    *agg_obj;
            TenObj    *ten_obj;
            uint64_t   ptr;
        };

        class BoolType;
        class FWidthType;
        class IntType;
        class FloatType;
        class StrType;
        class ArrType;
        class AggType;

        class ProgramHeap
        {
            std::vector<BoolType*>   bool_types;
            std::vector<FWidthType*> fwidth_types;
            std::vector<IntType*>    int_types;
            std::vector<FloatType*>  float_types;
            std::vector<StrType*>    str_types;
            std::vector<ArrType*>    arr_types;
            std::vector<AggType*>    agg_types;

            std::vector<BoolObj*>    bool_objs;
            std::vector<FWidthObj*>  fwidth_objs;
            std::vector<IntObj*>     int_objs;
            std::vector<FloatObj*>   float_objs;
            std::vector<StrObj*>     str_objs;
            std::vector<AggObj*>     agg_objs;
            std::vector<TenObj*>     ten_objs;

        public:
            ~ProgramHeap();

            bool create_type_bool   (Errors& errs, Obj& obj);
            bool create_type_fwidth (Errors& errs, Obj& obj);
            bool create_type_int    (Errors& errs, Obj& obj);
            bool create_type_float  (Errors& errs, Obj& obj);
            bool create_type_str    (Errors& errs, Obj& obj);
            bool create_type_arr    (Errors& errs, Obj& obj, TypeObj* ty);
            bool create_type_agg    (Errors& errs, Obj& obj, std::vector<TypeObj*> tys);

            bool create_obj_bool    (Errors& errs, Obj& obj, BoolObj val);
            bool create_obj_fwidth  (Errors& errs, Obj& obj, FWidthObj val);
            bool create_obj_int     (Errors& errs, Obj& obj, IntObj val);
            bool create_obj_float   (Errors& errs, Obj& obj, FloatObj val);
            bool create_obj_str     (Errors& errs, Obj& obj, const StrObj& val);
            bool create_obj_agg     (Errors& errs, Obj& obj, const AggObj& val);
            bool create_obj_tensor  (Errors& errs, Obj& obj, FWidthObj dty, const std::vector<IntObj>& dims);
        };

        class CallStack;

        class TypeObj
        {
        protected:
            ~TypeObj() {}
        public:
            virtual bool cpy  (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src) = 0;
            virtual bool inst (Errors& errs, ProgramHeap& heap, Obj& dst) = 0;
            virtual bool set  (Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs) = 0;
            virtual bool iadd (Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs);
            virtual bool isub (Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs);
            virtual bool imul (Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs);
            virtual bool idiv (Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs);
            virtual bool imod (Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs);
            virtual bool add  (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool sub  (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool mul  (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool div  (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool mod  (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool eq   (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool ne   (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool ge   (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool le   (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool gt   (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool lt   (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool land (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool lor  (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool idx  (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool xstr (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src);
            virtual bool xflt (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src);
            virtual bool xint (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src);
        };

        class BoolType :
            public TypeObj
        {
            friend bool ProgramHeap::create_type_bool(Errors& errs, Obj& obj);
            BoolType() = default;
        public:
            virtual bool cpy  (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool inst (Errors& errs, ProgramHeap& heap, Obj& dst) override;
            virtual bool set  (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool eq   (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool ne   (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool land (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool lor  (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool xstr (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src) override;
        };

        class FWidthType :
            public TypeObj
        {
            friend bool ProgramHeap::create_type_fwidth(Errors& errs, Obj& obj);
            FWidthType() = default;
        public:
            virtual bool cpy  (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src)  override;
            virtual bool inst (Errors& errs, ProgramHeap& heap, Obj& dst) override;
            virtual bool set  (Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool eq   (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool ne   (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool xstr (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src) override;
        };

        class IntType :
            public TypeObj
        {
            friend bool ProgramHeap::create_type_int(Errors& errs, Obj& obj);
            IntType() = default;
        public:
            virtual bool cpy  (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src)  override;
            virtual bool inst (Errors& errs, ProgramHeap& heap, Obj& dst) override;
            virtual bool set  (Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool iadd (Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool isub (Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool imul (Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool idiv (Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool imod (Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool add  (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool sub  (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool mul  (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool div  (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool mod  (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool eq   (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool ne   (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool ge   (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool le   (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool gt   (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool lt   (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool xstr (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool xflt (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool xint (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src) override;
        };

        class FloatType :
            public TypeObj
        {
            friend bool ProgramHeap::create_type_float(Errors& errs, Obj& obj);
            FloatType() = default;
        public:
            virtual bool cpy  (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src)  override;
            virtual bool inst (Errors& errs, ProgramHeap& heap, Obj& dst) override;
            virtual bool set  (Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool iadd (Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool isub (Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool imul (Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool idiv (Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool add  (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool sub  (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool mul  (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool div  (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool eq   (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool ne   (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool ge   (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool le   (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool gt   (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool lt   (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool xstr (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool xflt (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool xint (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src) override;
        };

        class StrType :
            public TypeObj
        {
            friend bool ProgramHeap::create_type_str(Errors& errs, Obj& obj);
            StrType() = default;
        public:
            virtual bool cpy  (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src)  override;
            virtual bool inst (Errors& errs, ProgramHeap& heap, Obj& dst) override;
            virtual bool set  (Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool iadd (Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool add  (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool eq   (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool ne   (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool ge   (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool le   (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool gt   (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool lt   (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool idx  (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool xstr (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool xflt (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool xint (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src) override;
        };

        class ArrType :  // arrays only.  inst -> agg
            public TypeObj
        {
            TypeObj* elem_ty;

            friend bool ProgramHeap::create_type_arr(Errors& errs, Obj& obj, TypeObj* ty);
            ArrType() = default;
        public:
            virtual bool cpy  (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool inst (Errors& errs, ProgramHeap& heap, Obj& dst) override;
            virtual bool set  (Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool iadd (Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool add  (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool eq   (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool ne   (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool idx  (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool xstr (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src) override;
        };

        class AggType :  // Tuples and structs.  inst -> agg
            public TypeObj
        {
            std::vector<TypeObj*> elem_tys;

            friend bool ProgramHeap::create_type_agg(Errors& errs, Obj& obj, std::vector<TypeObj*> tys);
            AggType() = default;
        public:
            virtual bool cpy  (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src)  override;
            virtual bool inst (Errors& errs, ProgramHeap& heap, Obj& dst) override;
            virtual bool set  (Errors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool eq   (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool ne   (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool idx  (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool xstr (Errors& errs, ProgramHeap& heap, Obj& dst, const Obj src) override;
        };
    }
}

#endif
