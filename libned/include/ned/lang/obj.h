#ifndef NED_OBJ_H
#define NED_OBJ_H

#include <ned/lang/ast.h>
#include <ned/core/graph.h>
#include <ned/core/config.h>

#include <string>
#include <sstream>
#include <unordered_map>

namespace nn
{
    namespace lang
    {
        union Obj;

        class TypeObj;
        using BoolObj = bool;
        using FtyObj = core::EdgeFty;
        using IntObj = int64_t;
        using FloatObj = double;
        using StrObj = std::string;
        using AggObj = std::vector<Obj>;
        using CfgObj = core::Config;

        union Obj
        {
            TypeObj   *type_obj;
            BoolObj   *bool_obj;
            FtyObj    *fty_obj;
            IntObj    *int_obj;
            FloatObj  *float_obj;
            StrObj    *str_obj;
            AggObj    *agg_obj;
            CfgObj    *cfg_obj;
            uint64_t   ptr;  // this field is use for code pointers, data pointers, and all graph stuff
        };

        class BoolType;
        class FtyType;
        class IntType;
        class FloatType;
        class StrType;
        class ArrType;
        class AggType;
        class CfgType;

        class ProgramHeap
        {
            std::vector<BoolType*>  bool_types;
            std::vector<FtyType*>   fty_types;
            std::vector<IntType*>   int_types;
            std::vector<FloatType*> float_types;
            std::vector<StrType*>   str_types;
            std::vector<ArrType*>   arr_types;
            std::vector<AggType*>   agg_types;
            std::vector<CfgType*>   cfg_types;

            std::vector<BoolObj*>   bool_objs;
            std::vector<FtyObj*>    fty_objs;
            std::vector<IntObj*>    int_objs;
            std::vector<FloatObj*>  float_objs;
            std::vector<StrObj*>    str_objs;
            std::vector<AggObj*>    agg_objs;
            std::vector<CfgObj*>    cfg_objs;
            // The program heap isn't responsible for managing the deep learning stuff.
            // That responsibility falls on the graph builder.

        public:
            ~ProgramHeap();

            bool create_type_bool  (Obj& obj);
            bool create_type_fty   (Obj& obj);
            bool create_type_int   (Obj& obj);
            bool create_type_float (Obj& obj);
            bool create_type_str   (Obj& obj);
            bool create_type_arr   (Obj& obj, TypeObj* ty);
            bool create_type_agg   (Obj& obj, std::vector<TypeObj*> tys);
            bool create_type_cfg   (Obj& obj);

            bool create_obj_bool   (Obj& obj, BoolObj val);
            bool create_obj_fty    (Obj& obj, FtyObj val);
            bool create_obj_int    (Obj& obj, IntObj val);
            bool create_obj_float  (Obj& obj, FloatObj val);
            bool create_obj_str    (Obj& obj, const StrObj& val);
            bool create_obj_agg    (Obj& obj, const AggObj& val);
            bool create_obj_cfg    (Obj& obj, const CfgObj& val);
        };

        class CallStack;

        class TypeObj
        {
        protected:
            ~TypeObj() {}
        public:
            virtual bool cpy  (ProgramHeap& heap, Obj& dst, const Obj src) = 0;
            virtual bool inst (ProgramHeap& heap, Obj& dst) = 0;
            virtual bool set  (ProgramHeap& heap, Obj& lhs, const Obj rhs) = 0;
            virtual bool iadd (ProgramHeap& heap, Obj& lhs, const Obj rhs);
            virtual bool isub (ProgramHeap& heap, Obj& lhs, const Obj rhs);
            virtual bool imul (ProgramHeap& heap, Obj& lhs, const Obj rhs);
            virtual bool idiv (ProgramHeap& heap, Obj& lhs, const Obj rhs);
            virtual bool imod (ProgramHeap& heap, Obj& lhs, const Obj rhs);
            virtual bool ipow (ProgramHeap& heap, Obj& lhs, const Obj rhs);
            virtual bool add  (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool sub  (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool mul  (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool div  (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool mod  (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool pow  (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool eq   (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool ne   (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool ge   (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool le   (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool gt   (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool lt   (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool land (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool lor  (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool idx  (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs);
            virtual bool len  (ProgramHeap& heap, Obj& dst, const Obj src);
            virtual bool neg  (ProgramHeap& heap, Obj& dst, const Obj src);
            virtual bool xcfg(ProgramHeap& heap, Obj& dst, const Obj src) = 0;
            virtual bool xstr (ProgramHeap& heap, Obj& dst, const Obj src);
            virtual bool xflt (ProgramHeap& heap, Obj& dst, const Obj src);
            virtual bool xint (ProgramHeap& heap, Obj& dst, const Obj src);
        };

        class BoolType :
            public TypeObj
        {
            friend bool ProgramHeap::create_type_bool(Obj& obj);
            BoolType() = default;
        public:
            virtual bool cpy  (ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool inst (ProgramHeap& heap, Obj& dst) override;
            virtual bool set  (ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool eq   (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool ne   (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool land (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool lor  (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool xcfg (ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool xstr (ProgramHeap& heap, Obj& dst, const Obj src) override;
        };

        class FtyType :
            public TypeObj
        {
            friend bool ProgramHeap::create_type_fty(Obj& obj);
            FtyType() = default;
        public:
            virtual bool cpy  (ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool inst (ProgramHeap& heap, Obj& dst) override;
            virtual bool set  (ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool eq   (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool ne   (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool xcfg (ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool xstr (ProgramHeap& heap, Obj& dst, const Obj src) override;
        };

        class IntType :
            public TypeObj
        {
            friend bool ProgramHeap::create_type_int(Obj& obj);
            IntType() = default;
        public:
            virtual bool cpy  (ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool inst (ProgramHeap& heap, Obj& dst) override;
            virtual bool set  (ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool iadd (ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool isub (ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool imul (ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool idiv (ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool imod (ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool ipow (ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool add  (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool sub  (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool mul  (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool div  (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool mod  (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool pow  (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool eq   (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool ne   (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool ge   (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool le   (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool gt   (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool lt   (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool neg  (ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool xcfg (ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool xstr (ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool xflt (ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool xint (ProgramHeap& heap, Obj& dst, const Obj src) override;
        };

        class FloatType :
            public TypeObj
        {
            friend bool ProgramHeap::create_type_float(Obj& obj);
            FloatType() = default;
        public:
            virtual bool cpy  (ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool inst (ProgramHeap& heap, Obj& dst) override;
            virtual bool set  (ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool iadd (ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool isub (ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool imul (ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool idiv (ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool ipow (ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool add  (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool sub  (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool mul  (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool div  (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool pow  (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool eq   (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool ne   (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool ge   (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool le   (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool gt   (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool lt   (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool neg  (ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool xcfg (ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool xstr (ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool xflt (ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool xint (ProgramHeap& heap, Obj& dst, const Obj src) override;
        };

        class StrType :
            public TypeObj
        {
            friend bool ProgramHeap::create_type_str(Obj& obj);
            StrType() = default;
        public:
            virtual bool cpy  (ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool inst (ProgramHeap& heap, Obj& dst) override;
            virtual bool set  (ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool iadd (ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool add  (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool eq   (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool ne   (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool ge   (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool le   (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool gt   (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool lt   (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool idx  (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool xcfg (ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool xstr (ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool xflt (ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool xint (ProgramHeap& heap, Obj& dst, const Obj src) override;
        };

        class ArrType :  // arrays only.  inst -> agg
            public TypeObj
        {
            TypeObj* elem_ty;

            friend bool ProgramHeap::create_type_arr(Obj& obj, TypeObj* ty);
            ArrType() = default;
        public:
            virtual bool cpy  (ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool inst (ProgramHeap& heap, Obj& dst) override;
            virtual bool set  (ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool iadd (ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool add  (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool eq   (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool ne   (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool idx  (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool len  (ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool xcfg (ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool xstr (ProgramHeap& heap, Obj& dst, const Obj src) override;
        };

        class AggType :  // Tuples and structs.  inst -> agg
            public TypeObj
        {
            std::vector<TypeObj*> elem_tys;

            friend bool ProgramHeap::create_type_agg(Obj& obj, std::vector<TypeObj*> tys);
            AggType() = default;
        public:
            virtual bool cpy  (ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool inst (ProgramHeap& heap, Obj& dst) override;
            virtual bool set  (ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool eq   (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool ne   (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool idx  (ProgramHeap& heap, Obj& dst, const Obj lhs, const Obj rhs) override;
            virtual bool len  (ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool xcfg (ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool xstr (ProgramHeap& heap, Obj& dst, const Obj src) override;
        };

        class CfgType :
            public TypeObj
        {
            friend bool ProgramHeap::create_type_cfg(Obj& obj);
            CfgType() = default;
        public:
            virtual bool cpy  (ProgramHeap& heap, Obj& dst, const Obj src) override;
            virtual bool inst (ProgramHeap& heap, Obj& dst) override;
            virtual bool set  (ProgramHeap& heap, Obj& lhs, const Obj rhs) override;
            virtual bool xcfg (ProgramHeap& heap, Obj& dst, const Obj src) override;
        };
    }
}

#endif
