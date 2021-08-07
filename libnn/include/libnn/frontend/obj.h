#ifndef NN_GENERATOR_H
#define NN_GENERATOR_H

#include <string>
#include <unordered_map>

#include <libnn/core/graph.h>
#include <libnn/frontend/ast.h>

namespace nn
{
    namespace impl
    {
        enum class ObjType
        {
            INVALID,  // Invalid object type
            BOOL,     // Boolean
            INT,      // Integer
            FLOAT,    // Floating point
            STR,      // String
            ARRAY,    // Array - Fixed length
            TUPLE,    // Tuple - mainly for cargs
            TENSOR,   // Tensor - Graph edge
            DEF,      // Non-intrinsic block
            FN,       // Generation time helper function
            INTR,     // Intrinsic block
            MODULE,   // *.nn file
            PACKAGE,  // File folder (presumably with *.nn files in it)
        };

        constexpr std::string objTypeName(ObjType ty);

        class Obj
        {
        public:
            ObjType ty = ObjType::INVALID;
            bool init;

            Obj(ObjType ty);
            virtual ~Obj() = 0;

            virtual bool bval() const = 0;
            virtual void assign(const Obj* val) = 0;

            virtual Obj* get(const std::string& item) = 0;
            virtual Obj* cargs(const std::vector<Obj*>& args) = 0;
            virtual Obj* call(EvalCtx& ctx, const std::vector<Obj*>& args) const = 0;
            virtual std::vector<Obj*> iter(EvalCtx& ctx) = 0;
            virtual Obj* idx(const std::vector<std::vector<Obj*>>& val) = 0;
            virtual Obj* neg() const = 0;

            virtual Obj* add(const Obj* val) const = 0;
            virtual Obj* sub(const Obj* val) const = 0;
            virtual Obj* mul(const Obj* val) const = 0;
            virtual Obj* div(const Obj* val) const = 0;

            // Why C++?  Wasn't && enough?
            virtual Obj* andop(const Obj* val) const = 0;
            virtual Obj* orop(const Obj* val) const = 0;

            virtual Obj* eq(const Obj* val) const = 0;
            virtual Obj* ne(const Obj* val) const = 0;
            virtual Obj* ge(const Obj* val) const = 0;
            virtual Obj* le(const Obj* val) const = 0;
            virtual Obj* gt(const Obj* val) const = 0;
            virtual Obj* lt(const Obj* val) const = 0;
        };

        template<ObjType TY>
        class ObjImp;

        using ObjBool = ObjImp<ObjType::BOOL>;
        using ObjInt = ObjImp<ObjType::INT>;
        using ObjFloat = ObjImp<ObjType::FLOAT>;
        using ObjStr = ObjImp<ObjType::STR>;
        using ObjTuple = ObjImp<ObjType::TUPLE>;
        using ObjTensor = ObjImp<ObjType::TENSOR>;
        using ObjDef = ObjImp<ObjType::DEF>;
        using ObjFn = ObjImp<ObjType::FN>;
        using ObjIntr = ObjImp<ObjType::INTR>;
        using ObjModule = ObjImp<ObjType::MODULE>;
        using ObjPackage = ObjImp<ObjType::PACKAGE>;

        template<ObjType TY>
        struct ObjData;

        template<ObjType TY>
        class ObjImp :
            public Obj
        {
            friend void check_init(const Obj*);
            friend void check_init(const Obj* pobj);

            auto mty(const Obj* p) const noexcept { return (const decltype(this))p; }
            auto mty(Obj* p) const noexcept { return static_cast<decltype(this)>(p); }

        public:
            ObjData<TY> data;

            ObjImp();

            virtual ~ObjImp();

            void check_type(const Obj* pobj) const {
                if (pobj->ty != TY) throw GenerationError("Expected " + objTypeName(TY) + ", recieved " + objTypeName(pobj->ty)); }

            virtual bool bval() const override {
                throw GenerationError(objTypeName(TY) + " type does not have a truth value"); }
            virtual void assign(const Obj* val) override {
                throw GenerationError(objTypeName(TY) + " type does not support assignment"); }
            virtual Obj* cargs(const std::vector<Obj*>& args) override {
                throw GenerationError(objTypeName(TY) + " type does not support constant arguments"); }

            virtual Obj* get(const std::string& item) override {
                throw GenerationError(objTypeName(TY) + " type does not support the get operator"); }
            virtual Obj* call(EvalCtx& ctx, const std::vector<Obj*>& args) const override {
                throw GenerationError(objTypeName(TY) + " type does not support the call operator"); }
            virtual std::vector<Obj*> iter(EvalCtx& ctx) override {
                throw GenerationError(objTypeName(TY) + " type does not support iteration"); }
            virtual Obj* idx(const std::vector<std::vector<Obj*>>& val) override {
                throw GenerationError(objTypeName(TY) + " type does not support the index operator"); }
            virtual Obj* neg() const override {
                throw GenerationError(objTypeName(TY) + " type does not support the negation operator"); }

            virtual Obj* add(const Obj* val) const override {
                throw GenerationError(objTypeName(TY) + " type does not support the addition operator"); }
            virtual Obj* sub(const Obj* val) const override {
                throw GenerationError(objTypeName(TY) + " type does not support the subtraction operator"); }
            virtual Obj* mul(const Obj* val) const override {
                throw GenerationError(objTypeName(TY) + " type does not support the multiplication operator"); }
            virtual Obj* div(const Obj* val) const override {
                throw GenerationError(objTypeName(TY) + " type does not support the division operator"); }

            virtual Obj* andop(const Obj* val) const override  {
                throw GenerationError(objTypeName(TY) + " type does not support the and operator"); }
            virtual Obj* orop(const Obj* val) const override  {
                throw GenerationError(objTypeName(TY) + " type does not support the or operator"); }

            virtual Obj* eq(const Obj* val) const override {
                throw GenerationError(objTypeName(TY) + " type does not support the equality operator"); }
            virtual Obj* ne(const Obj* val) const override {
                throw GenerationError(objTypeName(TY) + " type does not support the inequality operator"); }
            virtual Obj* ge(const Obj* val) const override {
                throw GenerationError(objTypeName(TY) + " type does not support the greater than or equal operator"); }
            virtual Obj* le(const Obj* val) const override {
                throw GenerationError(objTypeName(TY) + " type does not support the less than or equal operator"); }
            virtual Obj* gt(const Obj* val) const override {
                throw GenerationError(objTypeName(TY) + " type does not support the greater than operator"); }
            virtual Obj* lt(const Obj* val) const override {
                throw GenerationError(objTypeName(TY) + " type does not support the less than operator"); }
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
            std::vector<Obj*> elems;
            int size;
        };
        template<> struct ObjData<ObjType::TUPLE> {
            std::vector<Obj*> elems;
        };
        template<> struct ObjData<ObjType::TENSOR> {
            Edge* pEdge;
            std::vector<uint32_t> dims;
            bool carg_init;
        };
        template<> struct ObjData<ObjType::DEF> {
            AstDef def;
        };
        template<> struct ObjData<ObjType::FN> {
            AstFn fn;
        };
        template<> struct ObjData<ObjType::INTR> {
            AstIntr intr;
        };
        template<> struct ObjData<ObjType::MODULE> {
            AstModule mod;
        };
        template<> struct ObjData<ObjType::PACKAGE> {
            std::unordered_map<std::string, ObjModule> mods;
            std::unordered_map<std::string, ObjPackage> packs;
        };
    }
}

#endif
