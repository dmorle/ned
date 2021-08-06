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
            virtual void assign(const std::unique_ptr<Obj> val) = 0;

            virtual std::unique_ptr<Obj> get(const std::string& item) = 0;
            virtual std::unique_ptr<Obj> cargs(const std::vector<std::unique_ptr<Obj>> args) const = 0;
            virtual std::unique_ptr<Obj> call(EvalCtx& ctx, const std::vector<std::unique_ptr<Obj>> args) const = 0;
            virtual std::vector<std::unique_ptr<Obj>> iter(EvalCtx& ctx) = 0;
            virtual std::unique_ptr<Obj> idx(const std::vector<std::vector<std::unique_ptr<Obj>>> val) = 0;
            virtual std::unique_ptr<Obj> neg() = 0;

            virtual std::unique_ptr<Obj> add(const std::unique_ptr<Obj>& val) = 0;
            virtual std::unique_ptr<Obj> sub(const std::unique_ptr<Obj>& val) = 0;
            virtual std::unique_ptr<Obj> mul(const std::unique_ptr<Obj>& val) = 0;
            virtual std::unique_ptr<Obj> div(const std::unique_ptr<Obj>& val) = 0;

            virtual std::unique_ptr<Obj> eq(const std::unique_ptr<Obj>& val) = 0;
            virtual std::unique_ptr<Obj> ne(const std::unique_ptr<Obj>& val) = 0;
            virtual std::unique_ptr<Obj> ge(const std::unique_ptr<Obj>& val) = 0;
            virtual std::unique_ptr<Obj> le(const std::unique_ptr<Obj>& val) = 0;
            virtual std::unique_ptr<Obj> gt(const std::unique_ptr<Obj>& val) = 0;
            virtual std::unique_ptr<Obj> lt(const std::unique_ptr<Obj>& val) = 0;
        };

        template<ObjType TY>
        struct ObjData;

        template<ObjType TY>
        class ObjImp :
            public Obj
        {
            auto mty(const std::unique_ptr<Obj>& p) noexcept { return (const decltype(this))p.get(); }
            auto mty(std::unique_ptr<Obj>& p) noexcept { return static_cast<decltype(this)>(p.get()); }

            static constexpr std::string type_name = objTypeName(TY);
            ObjData<TY> data;

        public:
            ObjImp(const std::vector<std::unique_ptr<Obj>>&);                      // constructor for standalone declaration
            ObjImp(const std::vector<std::unique_ptr<Obj>>&, const ObjData<TY>&);  // constructor for assignment declaration

            const ObjData<TY>& getData() const { return data; }

            virtual ~ObjImp();

            virtual bool bval() const override;
            virtual void assign(const std::unique_ptr<Obj> val) override;

            virtual std::unique_ptr<Obj> get(const std::string& item) override;
            virtual std::unique_ptr<Obj> cargs(const std::vector<std::unique_ptr<Obj>> args) const override;
            virtual std::unique_ptr<Obj> call(EvalCtx& ctx, const std::vector<std::unique_ptr<Obj>> args) const override;
            virtual std::unique_ptr<Obj> idx(const std::vector<std::vector<std::unique_ptr<Obj>>> val) override;
            virtual std::unique_ptr<Obj> neg() override;

            virtual std::unique_ptr<Obj> add(const std::unique_ptr<Obj>& val) override;
            virtual std::unique_ptr<Obj> sub(const std::unique_ptr<Obj>& val) override;
            virtual std::unique_ptr<Obj> mul(const std::unique_ptr<Obj>& val) override;
            virtual std::unique_ptr<Obj> div(const std::unique_ptr<Obj>& val) override;

            virtual std::unique_ptr<Obj> eq(const std::unique_ptr<Obj>& val) override;
            virtual std::unique_ptr<Obj> ne(const std::unique_ptr<Obj>& val) override;
            virtual std::unique_ptr<Obj> ge(const std::unique_ptr<Obj>& val) override;
            virtual std::unique_ptr<Obj> le(const std::unique_ptr<Obj>& val) override;
            virtual std::unique_ptr<Obj> gt(const std::unique_ptr<Obj>& val) override;
            virtual std::unique_ptr<Obj> lt(const std::unique_ptr<Obj>& val) override;
        };

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
        template<> struct ObjData<ObjType::TUPLE> {
            std::vector<std::string> val;
        };
        template<> struct ObjData<ObjType::TENSOR> {
            Edge* pEdge;
            std::vector<uint32_t> dims;
            std::string graph_name;  // only initialized for input edges
        };
        template<> struct ObjData<ObjType::DEF> {
            AstDef def;
            std::map<std::string, std::unique_ptr<Obj>> cargs;
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
