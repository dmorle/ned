#ifndef NED_OBJ_H
#define NED_OBJ_H

#include <string>
#include <sstream>
#include <unordered_map>

#include <ned/core/graph.h>
#include <ned/lang/ast.h>

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
            void*      ptr;
        };

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

            bool create_type_bool   (RuntimeErrors& errs, Obj& obj);
            bool create_type_fwidth (RuntimeErrors& errs, Obj& obj);
            bool create_type_int    (RuntimeErrors& errs, Obj& obj);
            bool create_type_float  (RuntimeErrors& errs, Obj& obj);
            bool create_type_str    (RuntimeErrors& errs, Obj& obj);
            bool create_type_arr    (RuntimeErrors& errs, Obj& obj, TypeObj* ty);
            bool create_type_agg    (RuntimeErrors& errs, Obj& obj, std::vector<TypeObj*> tys);

            bool create_obj_bool    (RuntimeErrors& errs, Obj& obj, BoolObj val);
            bool create_obj_fwidth  (RuntimeErrors& errs, Obj& obj, FWidthObj val);
            bool create_obj_int     (RuntimeErrors& errs, Obj& obj, IntObj val);
            bool create_obj_float   (RuntimeErrors& errs, Obj& obj, FloatObj val);
            bool create_obj_str     (RuntimeErrors& errs, Obj& obj, const StrObj& val);
            bool create_obj_agg     (RuntimeErrors& errs, Obj& obj, const AggObj& val);
            bool create_obj_tensor  (RuntimeErrors& errs, Obj& obj, FWidthObj dty, const std::vector<IntObj>& dims);
        };

        class CallStack;

        class TypeObj
        {
        protected:
            ~TypeObj();
        public:
            virtual bool cpy  (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src) = 0;
            virtual bool inst (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst) = 0;
            virtual bool set  (RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs) = 0;
            virtual bool iadd (RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs);
            virtual bool isub (RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs);
            virtual bool imul (RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs);
            virtual bool idiv (RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs);
            virtual bool imod (RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs);
            virtual bool add  (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool sub  (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool mul  (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool div  (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool mod  (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool eq   (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool ne   (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool ge   (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool le   (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool gt   (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool lt   (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool land (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool lor  (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool idx  (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool xstr (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src);
            virtual bool xflt (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src);
            virtual bool xint (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src);
        };

        class BoolType :
            public TypeObj
        {
            friend bool ProgramHeap::create_type_bool(RuntimeErrors& errs, Obj& obj);
            BoolType() = default;
        public:
            virtual bool cpy  (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool inst (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst) override;
            virtual bool set  (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool eq   (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool ne   (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool land (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool lor  (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool xstr (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src) override;
        };

        class FWidthType :
            public TypeObj
        {
            friend bool ProgramHeap::create_type_fwidth(RuntimeErrors& errs, Obj& obj);
            FWidthType() = default;
        public:
            virtual bool cpy  (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src)  override;
            virtual bool inst (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst) override;
            virtual bool set  (RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool eq   (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool ne   (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool xstr (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src) override;
        };

        class IntType :
            public TypeObj
        {
            friend bool ProgramHeap::create_type_int(RuntimeErrors& errs, Obj& obj);
            IntType() = default;
        public:
            virtual bool cpy  (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src)  override;
            virtual bool inst (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst) override;
            virtual bool set  (RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool iadd (RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool isub (RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool imul (RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool idiv (RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool imod (RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool add  (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool sub  (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool mul  (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool div  (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool mod  (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool eq   (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool ne   (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool ge   (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool le   (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool gt   (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool lt   (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool xstr (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool xflt (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool xint (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src) override;
        };

        class FloatType :
            public TypeObj
        {
            friend bool ProgramHeap::create_type_float(RuntimeErrors& errs, Obj& obj);
            FloatType() = default;
        public:
            virtual bool cpy  (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src)  override;
            virtual bool inst (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst) override;
            virtual bool set  (RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool iadd (RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool isub (RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool imul (RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool idiv (RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool add  (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool sub  (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool mul  (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool div  (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool eq   (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool ne   (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool ge   (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool le   (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool gt   (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool lt   (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool xstr (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool xflt (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool xint (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src) override;
        };

        class StrType :
            public TypeObj
        {
            friend bool ProgramHeap::create_type_str(RuntimeErrors& errs, Obj& obj);
            StrType() = default;
        public:
            virtual bool cpy  (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src)  override;
            virtual bool inst (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst) override;
            virtual bool set  (RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool iadd (RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool add  (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool eq   (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool ne   (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool ge   (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool le   (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool gt   (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool lt   (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool idx  (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool xstr (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool xflt (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool xint (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src) override;
        };

        class ArrType :  // arrays only.  inst -> agg
            public TypeObj
        {
            TypeObj* elem_ty;

            friend bool ProgramHeap::create_type_arr(RuntimeErrors& errs, Obj& obj, TypeObj* ty);
            ArrType() = default;
        public:
            virtual bool cpy  (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool inst (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst) override;
            virtual bool set  (RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool iadd (RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool add  (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool eq   (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool ne   (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool idx  (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool xstr (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src) override;
        };

        class AggType :  // Tuples and structs.  inst -> agg
            public TypeObj
        {
            std::vector<TypeObj*> elem_tys;

            friend bool ProgramHeap::create_type_agg(RuntimeErrors& errs, Obj& obj, std::vector<TypeObj*> tys);
            AggType() = default;
        public:
            virtual bool cpy  (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src)  override;
            virtual bool inst (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst) override;
            virtual bool set  (RuntimeErrors& errs, ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool eq   (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool ne   (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool idx  (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool xstr (RuntimeErrors& errs, ProgramHeap& heap, Obj& dst, const Obj src) override;
        };
    }
}

#endif
