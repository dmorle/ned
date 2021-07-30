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
            INTR,     // Intrinsic block
            DEF,      // Non-intrinsic block
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
            virtual Obj* copy() const = 0;

            virtual Obj* get(const std::string& item) = 0;
            virtual Obj* cargs(const std::vector<Obj*> args) = 0;
            virtual Obj* call(const std::vector<Obj*> args) = 0;
            virtual Obj* idx(const std::vector<std::vector<Obj*>> val) = 0;
            virtual Obj* neg() = 0;

            virtual Obj* add(const Obj* val) = 0;
            virtual Obj* sub(const Obj* val) = 0;
            virtual Obj* mul(const Obj* val) = 0;
            virtual Obj* div(const Obj* val) = 0;

            virtual Obj* eq(const Obj* val) = 0;
            virtual Obj* ne(const Obj* val) = 0;
            virtual Obj* ge(const Obj* val) = 0;
            virtual Obj* le(const Obj* val) = 0;
            virtual Obj* gt(const Obj* val) = 0;
            virtual Obj* lt(const Obj* val) = 0;
        };

        template<ObjType TY>
        struct ObjData;

        template<ObjType TY>
        class ObjImp :
            public Obj
        {
            auto mty(const Obj* p) noexcept { return (const decltype(this))p; }
            auto mty(Obj* p) noexcept { return static_cast<decltype(this)>(p); }

            static constexpr std::string type_name = objTypeName(TY);
            ObjData<TY> data;

        public:
            ObjImp() : Obj(TY) {}
            virtual ~ObjImp();

            virtual bool bval() const override;
            virtual void assign(const Obj* val) override;
            virtual Obj* copy() const override;

            virtual Obj* get(const std::string& item);
            virtual Obj* neg();
            virtual Obj* call(const std::vector<Obj*> args);
            virtual Obj* idx(const std::vector<std::vector<Obj*>> val);

            virtual Obj* add(const Obj* val);
            virtual Obj* sub(const Obj* val);
            virtual Obj* mul(const Obj* val);
            virtual Obj* div(const Obj* val);

            virtual Obj* eq(const Obj* val);
            virtual Obj* ne(const Obj* val);
            virtual Obj* ge(const Obj* val);
            virtual Obj* le(const Obj* val);
            virtual Obj* gt(const Obj* val);
            virtual Obj* lt(const Obj* val);
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
        template<> struct ObjData<ObjType::TUPLE> {
            std::vector<std::string> val;
        };
        template<> struct ObjData<ObjType::TENSOR> {
            Edge* pEdge;
            std::vector<uint32_t> dims;
        };
    }
}

#endif
