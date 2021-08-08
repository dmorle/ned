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
            virtual void assign(const std::shared_ptr<Obj>& val) = 0;

            virtual std::shared_ptr<Obj> get(const std::string& item) = 0;
            virtual std::shared_ptr<Obj> cargs(const std::vector<std::shared_ptr<Obj>>& args) = 0;
            virtual std::shared_ptr<Obj> call(EvalCtx& ctx, const std::vector<std::shared_ptr<Obj>>& args) const = 0;
            virtual std::vector<std::shared_ptr<Obj>> iter(EvalCtx& ctx) = 0;
            virtual std::shared_ptr<Obj> idx(const std::vector<std::vector<std::shared_ptr<Obj>>>& val) = 0;
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

        using ObjInvalid = ObjImp<ObjType::INVALID>;
        using ObjBool = ObjImp<ObjType::BOOL>;
        using ObjInt = ObjImp<ObjType::INT>;
        using ObjFloat = ObjImp<ObjType::FLOAT>;
        using ObjStr = ObjImp<ObjType::STR>;
        using ObjArray = ObjImp<ObjType::ARRAY>;
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
            friend void check_init(const std::shared_ptr<Obj>& pobj);

            template<ObjType T, typename... Args>
            friend std::shared_ptr<ObjImp<T>> create_obj(Args...);

            auto mty(const std::shared_ptr<Obj>& p) const noexcept { return (const decltype(this))p.get(); }
            auto mty(std::shared_ptr<Obj> &p) const noexcept { return static_cast<decltype(this)>(p.get()); }

            ObjImp();

        public:
            ObjData<TY> data;

            virtual ~ObjImp();

            void check_type(const std::shared_ptr<Obj>& pobj) const {
                if (pobj->ty != TY) throw GenerationError("Expected " + objTypeName(TY) + ", recieved " + objTypeName(pobj->ty)); }

            virtual bool bval() const override {
                throw GenerationError(objTypeName(TY) + " type does not have a truth value"); }
            virtual void assign(const std::shared_ptr<Obj>& val) override {
                throw GenerationError(objTypeName(TY) + " type does not support assignment"); }

            virtual std::shared_ptr<Obj> get(const std::string& item) override {
                throw GenerationError(objTypeName(TY) + " type does not support the get operator"); }
            virtual std::shared_ptr<Obj> call(EvalCtx& ctx, const std::vector<std::shared_ptr<Obj>>& args) const override {
                throw GenerationError(objTypeName(TY) + " type does not support the call operator"); }
            virtual std::shared_ptr<Obj> cargs(const std::vector<std::shared_ptr<Obj>>& args) override {
                throw GenerationError(objTypeName(TY) + " type does not support constant arguments"); }
            virtual std::vector<std::shared_ptr<Obj>> iter(EvalCtx& ctx) override {
                throw GenerationError(objTypeName(TY) + " type does not support iteration"); }
            virtual std::shared_ptr<Obj> idx(const std::vector<std::vector<std::shared_ptr<Obj>>>& val) override {
                throw GenerationError(objTypeName(TY) + " type does not support the index operator"); }
            virtual std::shared_ptr<Obj> neg() const override {
                throw GenerationError(objTypeName(TY) + " type does not support the negation operator"); }

            virtual std::shared_ptr<Obj> add(const std::shared_ptr<Obj>& val) const override {
                throw GenerationError(objTypeName(TY) + " type does not support the addition operator"); }
            virtual std::shared_ptr<Obj> sub(const std::shared_ptr<Obj>& val) const override {
                throw GenerationError(objTypeName(TY) + " type does not support the subtraction operator"); }
            virtual std::shared_ptr<Obj> mul(const std::shared_ptr<Obj>& val) const override {
                throw GenerationError(objTypeName(TY) + " type does not support the multiplication operator"); }
            virtual std::shared_ptr<Obj> div(const std::shared_ptr<Obj>& val) const override {
                throw GenerationError(objTypeName(TY) + " type does not support the division operator"); }

            virtual std::shared_ptr<Obj> andop(const std::shared_ptr<Obj>& val) const override  {
                throw GenerationError(objTypeName(TY) + " type does not support the and operator"); }
            virtual std::shared_ptr<Obj> orop(const std::shared_ptr<Obj>& val) const override  {
                throw GenerationError(objTypeName(TY) + " type does not support the or operator"); }

            virtual std::shared_ptr<Obj> eq(const std::shared_ptr<Obj>& val) const override {
                throw GenerationError(objTypeName(TY) + " type does not support the equality operator"); }
            virtual std::shared_ptr<Obj> ne(const std::shared_ptr<Obj>& val) const override {
                throw GenerationError(objTypeName(TY) + " type does not support the inequality operator"); }
            virtual std::shared_ptr<Obj> ge(const std::shared_ptr<Obj>& val) const override {
                throw GenerationError(objTypeName(TY) + " type does not support the greater than or equal operator"); }
            virtual std::shared_ptr<Obj> le(const std::shared_ptr<Obj>& val) const override {
                throw GenerationError(objTypeName(TY) + " type does not support the less than or equal operator"); }
            virtual std::shared_ptr<Obj> gt(const std::shared_ptr<Obj>& val) const override {
                throw GenerationError(objTypeName(TY) + " type does not support the greater than operator"); }
            virtual std::shared_ptr<Obj> lt(const std::shared_ptr<Obj>& val) const override {
                throw GenerationError(objTypeName(TY) + " type does not support the less than operator"); }
        };

        template<> struct ObjData<ObjType::INVALID> {};
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
            int size;
        };
        template<> struct ObjData<ObjType::TUPLE> {
            std::vector<std::shared_ptr<Obj>> elems;
        };
        template<> struct ObjData<ObjType::TENSOR> {
            Edge* pEdge;
            std::vector<uint32_t> dims;
            bool carg_init;
        };
        template<> struct ObjData<ObjType::DEF> {
            const AstDef* pdef;
            Scope* pscope;  // non-null for .cargs()
        };
        template<> struct ObjData<ObjType::FN> {
            const AstFn* pfn;
        };
        template<> struct ObjData<ObjType::INTR> {
            const AstIntr* pintr;
            Scope* pscope;
        };
        template<> struct ObjData<ObjType::MODULE> {
            AstModule mod;
        };
        template<> struct ObjData<ObjType::PACKAGE> {
            std::unordered_map<std::string, ObjModule> mods;
            std::unordered_map<std::string, ObjPackage> packs;
        };

        template<ObjType T, typename... Args>
        std::shared_ptr<ObjImp<T>> create_obj(Args...);

        template<> std::shared_ptr<ObjInvalid> create_obj<ObjType::INVALID>();
        template<> std::shared_ptr<ObjBool> create_obj<ObjType::BOOL>();
        template<> std::shared_ptr<ObjBool> create_obj<ObjType::BOOL, bool>(bool val);
        template<> std::shared_ptr<ObjInt> create_obj<ObjType::INT>();
        template<> std::shared_ptr<ObjInt> create_obj<ObjType::INT, int64_t>(int64_t val);
        template<> std::shared_ptr<ObjFloat> create_obj<ObjType::FLOAT>();
        template<> std::shared_ptr<ObjFloat> create_obj<ObjType::FLOAT, double>(double val);
        template<> std::shared_ptr<ObjStr> create_obj<ObjType::STR>();
        template<> std::shared_ptr<ObjStr> create_obj<ObjType::STR, const std::string&>(const std::string& val);
        template<> std::shared_ptr<ObjArray> create_obj<ObjType::ARRAY>();
        template<> std::shared_ptr<ObjArray> create_obj<ObjType::ARRAY, size_t, ObjType>(size_t sz, ObjType ty);
        template<> std::shared_ptr<ObjTuple> create_obj<ObjType::TUPLE>();
        template<> std::shared_ptr<ObjTuple> create_obj<ObjType::TUPLE, const std::vector<ObjType>&>(const std::vector<ObjType>& elems);
        template<> std::shared_ptr<ObjTensor> create_obj<ObjType::TENSOR>();
        template<> std::shared_ptr<ObjDef> create_obj<ObjType::DEF, const AstDef*>(const AstDef* pdef);
        template<> std::shared_ptr<ObjFn> create_obj<ObjType::FN, const AstFn*>(const AstFn* pfn);
        template<> std::shared_ptr<ObjIntr> create_obj<ObjType::INTR, const AstIntr*>(const AstIntr* pintr);
    }
}

#endif
