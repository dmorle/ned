#ifndef NED_GENERATOR_H
#define NED_GENERATOR_H

#include <string>
#include <sstream>
#include <unordered_map>

#include <ned/core/graph.h>
#include <ned/lang/ast.h>

namespace nn
{
    namespace lang
    {
        enum class ObjType
        {
            INVALID,  // Invalid object type
            TYPE,     // Type object
            BOOL,     // Boolean
            FWIDTH,   // Float widths for tensors, ie f16, f32, f64
            INT,      // Integer
            FLOAT,    // Floating point
            STR,      // String
            ARRAY,    // Array - Fixed length
            TUPLE,    // Tuple - mainly for cargs
            TENSOR,   // Tensor - Graph edge
            STRUCT,   // Structure
            FN,       // Generation time helper function
            DEF,      // Non-intrinsic block
            INTR,     // Intrinsic block
            MODULE,   // *.nn file
            PACKAGE,  // File folder (presumably with *.nn files in it)
        };

        ObjType dec_typename_exc(const std::string&);
        ObjType dec_typename_inv(const std::string&) noexcept;
        constexpr std::string obj_type_name(ObjType ty);

        class Obj
        {
        public:
            ObjType ty = ObjType::INVALID;
            bool init;

            Obj(ObjType ty);
            virtual ~Obj() {}

            virtual bool bval() const = 0;
            virtual std::string str() const = 0;
            virtual void assign(const std::shared_ptr<Obj>& val) = 0;
            virtual bool check_cargs(const std::vector<std::shared_ptr<Obj>>& args) const = 0;
            virtual std::shared_ptr<Obj> copy() const = 0;
            virtual std::shared_ptr<Obj> type() const = 0;
            virtual std::shared_ptr<Obj> inst() const = 0;

            virtual void call(EvalCtx& ctx, const std::vector<std::shared_ptr<Obj>>& args) const = 0;
            virtual std::shared_ptr<Obj> get(const std::string& item) = 0;
            virtual std::shared_ptr<Obj> cargs(const std::vector<std::shared_ptr<Obj>>& args) = 0;
            virtual std::vector<std::shared_ptr<Obj>> iter() = 0;
            virtual std::shared_ptr<Obj> idx(const std::shared_ptr<Obj>& val) = 0;
            virtual std::shared_ptr<Obj> neg() const = 0;

            virtual std::shared_ptr<Obj> add(const std::shared_ptr<Obj>& val) const = 0;
            virtual std::shared_ptr<Obj> sub(const std::shared_ptr<Obj>& val) const = 0;
            virtual std::shared_ptr<Obj> mul(const std::shared_ptr<Obj>& val) const = 0;
            virtual std::shared_ptr<Obj> div(const std::shared_ptr<Obj>& val) const = 0;

            // Why C++?  Wasn't && enough?
            virtual std::shared_ptr<Obj> andop(const std::shared_ptr<Obj>& val) const = 0;
            virtual std::shared_ptr<Obj> orop(const std::shared_ptr<Obj>& val) const = 0;

            virtual std::shared_ptr<Obj> eq(const std::shared_ptr<Obj>& val) const = 0;
            virtual std::shared_ptr<Obj> ne(const std::shared_ptr<Obj>& val) const = 0;
            virtual std::shared_ptr<Obj> ge(const std::shared_ptr<Obj>& val) const = 0;
            virtual std::shared_ptr<Obj> le(const std::shared_ptr<Obj>& val) const = 0;
            virtual std::shared_ptr<Obj> gt(const std::shared_ptr<Obj>& val) const = 0;
            virtual std::shared_ptr<Obj> lt(const std::shared_ptr<Obj>& val) const = 0;
        };

        template<ObjType TY>
        class ObjImp;

        using ObjDType   = ObjImp< ObjType::TYPE    >;
        using ObjInvalid = ObjImp< ObjType::INVALID >;
        using ObjVar     = ObjImp< ObjType::VAR     >;
        using ObjFWidth  = ObjImp< ObjType::FWIDTH  >;
        using ObjBool    = ObjImp< ObjType::BOOL    >;
        using ObjInt     = ObjImp< ObjType::INT     >;
        using ObjFloat   = ObjImp< ObjType::FLOAT   >;
        using ObjStr     = ObjImp< ObjType::STR     >;
        using ObjArray   = ObjImp< ObjType::ARRAY   >;
        using ObjTuple   = ObjImp< ObjType::TUPLE   >;
        using ObjTensor  = ObjImp< ObjType::TENSOR  >;
        using ObjDef     = ObjImp< ObjType::DEF     >;
        using ObjFn      = ObjImp< ObjType::FN      >;
        using ObjIntr    = ObjImp< ObjType::INTR    >;
        using ObjModule  = ObjImp< ObjType::MODULE  >;
        using ObjPackage = ObjImp< ObjType::PACKAGE >;

        template<ObjType TY>
        struct ObjData;

        template<ObjType TY>
        class ObjImp :
            public Obj
        {
        public:
            ObjImp();

            auto mty(const std::shared_ptr<Obj>& p) const noexcept { return (const decltype(this))p.get(); }
            auto mty(std::shared_ptr<Obj> &p) const noexcept { return static_cast<decltype(this)>(p.get()); }

            mutable ObjData<TY> data;

            virtual ~ObjImp();

            void check_mtype(const std::shared_ptr<Obj>& pobj) const {
                if (pobj->ty != TY) throw GenerationError("Expected " + obj_type_name(TY) + ", recieved " + obj_type_name(pobj->ty)); }

            virtual bool bval() const override {
                throw GenerationError(obj_type_name(TY) + " type does not have a truth value"); }
            virtual std::string str() const override {
                std::ostringstream addr; addr << (void*)this;
                return obj_type_name(TY) + " object at " + addr.str(); }
            virtual void assign(const std::shared_ptr<Obj>& val) override {
                throw GenerationError(obj_type_name(TY) + " type does not support assignment"); }
            virtual bool check_cargs(const std::vector<std::shared_ptr<Obj>>& args) const override { return true; }
            virtual std::shared_ptr<Obj> copy() const override {
                throw GenerationError(obj_type_name(TY) + " type does not support copying"); }
            virtual std::shared_ptr<Obj> type() const override { return create_obj_dtype(TY); }
            virtual std::shared_ptr<Obj> inst() const override {
                throw GenerationError(obj_type_name(TY) + " type can not be instantiated"); }

            virtual void call(EvalCtx& ctx, const std::vector<std::shared_ptr<Obj>>& args) const override {
                throw GenerationError(obj_type_name(TY) + " type does not support the call operator"); }
            virtual std::shared_ptr<Obj> get(const std::string& item) override {
                throw GenerationError(obj_type_name(TY) + " type does not support the get operator"); }
            virtual std::shared_ptr<Obj> cargs(const std::vector<std::shared_ptr<Obj>>& args) override {
                throw GenerationError(obj_type_name(TY) + " type does not support constant arguments"); }
            virtual std::vector<std::shared_ptr<Obj>> iter() override {
                throw GenerationError(obj_type_name(TY) + " type does not support iteration"); }
            virtual std::shared_ptr<Obj> idx(const std::shared_ptr<Obj>& val) override {
                throw GenerationError(obj_type_name(TY) + " type does not support the index operator"); }
            virtual std::shared_ptr<Obj> neg() const override {
                throw GenerationError(obj_type_name(TY) + " type does not support the negation operator"); }

            virtual std::shared_ptr<Obj> add(const std::shared_ptr<Obj>& val) const override {
                throw GenerationError(obj_type_name(TY) + " type does not support the addition operator"); }
            virtual std::shared_ptr<Obj> sub(const std::shared_ptr<Obj>& val) const override {
                throw GenerationError(obj_type_name(TY) + " type does not support the subtraction operator"); }
            virtual std::shared_ptr<Obj> mul(const std::shared_ptr<Obj>& val) const override {
                throw GenerationError(obj_type_name(TY) + " type does not support the multiplication operator"); }
            virtual std::shared_ptr<Obj> div(const std::shared_ptr<Obj>& val) const override {
                throw GenerationError(obj_type_name(TY) + " type does not support the division operator"); }

            virtual std::shared_ptr<Obj> andop(const std::shared_ptr<Obj>& val) const override  {
                throw GenerationError(obj_type_name(TY) + " type does not support the and operator"); }
            virtual std::shared_ptr<Obj> orop(const std::shared_ptr<Obj>& val) const override  {
                throw GenerationError(obj_type_name(TY) + " type does not support the or operator"); }

            virtual std::shared_ptr<Obj> eq(const std::shared_ptr<Obj>& val) const override {
                throw GenerationError(obj_type_name(TY) + " type does not support the equality operator"); }
            virtual std::shared_ptr<Obj> ne(const std::shared_ptr<Obj>& val) const override {
                throw GenerationError(obj_type_name(TY) + " type does not support the inequality operator"); }
            virtual std::shared_ptr<Obj> ge(const std::shared_ptr<Obj>& val) const override {
                throw GenerationError(obj_type_name(TY) + " type does not support the greater than or equal operator"); }
            virtual std::shared_ptr<Obj> le(const std::shared_ptr<Obj>& val) const override {
                throw GenerationError(obj_type_name(TY) + " type does not support the less than or equal operator"); }
            virtual std::shared_ptr<Obj> gt(const std::shared_ptr<Obj>& val) const override {
                throw GenerationError(obj_type_name(TY) + " type does not support the greater than operator"); }
            virtual std::shared_ptr<Obj> lt(const std::shared_ptr<Obj>& val) const override {
                throw GenerationError(obj_type_name(TY) + " type does not support the less than operator"); }
        };

        template<>
        struct ObjData<ObjType::TYPE>
        {
            ObjType ety;
            std::vector<std::shared_ptr<Obj>> cargs;
            bool has_cargs;
        };
        template<> struct ObjData<ObjType::INVALID> {};
        template<> struct ObjData<ObjType::VAR> {
            std::shared_ptr<Obj> self;
        };
        template<> struct ObjData<ObjType::FWIDTH> {
            core::tensor_dty dty;
        };
        template<> struct ObjData<ObjType::BOOL> {
            bool val;
        };
        template<> struct ObjData<ObjType::INT> {
            int64_t val;
        };
        template<> struct ObjData<ObjType::FLOAT> {
            double val;
        };
        template<> struct ObjData<ObjType::STR> {
            std::string val;
        };
        template<> struct ObjData<ObjType::ARRAY> {
            std::vector<std::shared_ptr<Obj>> elems;
            std::shared_ptr<Obj> dtype;
        };
        template<> struct ObjData<ObjType::TUPLE> {
            std::vector<std::shared_ptr<Obj>> elems;
            std::vector<std::shared_ptr<Obj>> dtypes;
        };
        template<> struct ObjData<ObjType::TENSOR> {
            mutable core::Edge* pEdge;
            std::vector<uint32_t> dims;
            core::tensor_dty dty;
            bool carg_init;
            bool is_static;
        };
        template<> struct ObjData<ObjType::DEF> {
            const AstDef* pdef;
            std::vector<std::shared_ptr<Obj>> cargs;
            bool has_cargs;
        };
        template<> struct ObjData<ObjType::FN> {
            const AstFn* pfn;
        };
        template<> struct ObjData<ObjType::INTR> {
            const AstIntr* pintr;
            std::vector<std::shared_ptr<Obj>> cargs;
            bool has_cargs;
        };
        template<> struct ObjData<ObjType::MODULE> {
            //AstModule mod;
        };
        template<> struct ObjData<ObjType::PACKAGE> {
            std::unordered_map<std::string, ObjModule> mods;
            std::unordered_map<std::string, ObjPackage> packs;
        };

        std::shared_ptr<Obj> create_obj_type(ObjType ty);
        std::shared_ptr< ObjDType   > create_obj_dtype   ();
        std::shared_ptr< ObjDType   > create_obj_dtype   (ObjType ty);
        std::shared_ptr< ObjDType   > create_obj_dtype   (ObjType ty, const std::vector<std::shared_ptr<Obj>>& cargs);
        std::shared_ptr< ObjInvalid > create_obj_invalid ();
        std::shared_ptr< ObjVar     > create_obj_var     ();
        std::shared_ptr< ObjFWidth  > create_obj_fwidth  ();
        std::shared_ptr< ObjFWidth  > create_obj_fwidth  (core::tensor_dty dty);
        std::shared_ptr< ObjBool    > create_obj_bool    ();
        std::shared_ptr< ObjBool    > create_obj_bool    (bool val);
        std::shared_ptr< ObjInt     > create_obj_int     ();
        std::shared_ptr< ObjInt     > create_obj_int     (int64_t val);
        std::shared_ptr< ObjFloat   > create_obj_float   ();
        std::shared_ptr< ObjFloat   > create_obj_float   (double val);
        std::shared_ptr< ObjStr     > create_obj_str     ();
        std::shared_ptr< ObjStr     > create_obj_str     (const std::string& val);
        std::shared_ptr< ObjArray   > create_obj_array   ();
        std::shared_ptr< ObjArray   > create_obj_array   (const std::shared_ptr<Obj> dtype, int sz);
        std::shared_ptr< ObjArray   > create_obj_array   (const std::shared_ptr<Obj> dtype, const std::vector<std::shared_ptr<Obj>>& elems);
        std::shared_ptr< ObjTuple   > create_obj_tuple   ();
        std::shared_ptr< ObjTuple   > create_obj_tuple   (const std::vector<std::shared_ptr<ObjDType>>& dtypes);
        std::shared_ptr< ObjTuple   > create_obj_tuple   (const std::vector<std::shared_ptr<Obj>>& elems);
        std::shared_ptr< ObjTensor  > create_obj_tensor  ();
        std::shared_ptr< ObjTensor  > create_obj_tensor  (core::tensor_dty dty, const std::vector<uint32_t>& dims);
        std::shared_ptr< ObjDef     > create_obj_def     ();
        std::shared_ptr< ObjDef     > create_obj_def     (const AstDef* pdef);
        std::shared_ptr< ObjDef     > create_obj_def     (const AstDef* pdef, const std::vector<std::shared_ptr<Obj>>& cargs);
        std::shared_ptr< ObjFn      > create_obj_fn      ();
        std::shared_ptr< ObjFn      > create_obj_fn      (const AstFn* pfn);
        std::shared_ptr< ObjIntr    > create_obj_intr    ();
        std::shared_ptr< ObjIntr    > create_obj_intr    (const AstIntr* pintr);
        std::shared_ptr< ObjIntr    > create_obj_intr    (const AstIntr* pintr, const std::vector<std::shared_ptr<Obj>>& cargs);
        std::shared_ptr< ObjModule  > create_obj_module  ();
        std::shared_ptr< ObjPackage > create_obj_package ();
    }
}

#endif
