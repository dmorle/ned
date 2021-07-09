#ifndef NN_GENERATOR_H
#define NN_GENERATOR_H

#include <string>
#include <unordered_map>

#include <libnn/core/graph.h>
#include <libnn/frontend/ast.h>

namespace nn
{
    namespace frontend
    {
        class Obj
        {
            size_t n_ref;
            bool is_valid;

        public:
            Obj();
            ~Obj();

            void incref() noexcept;
            void decref() noexcept;

            virtual Obj* add(const Obj* lval, const Obj* rval);
        };

        using EvalCtx = std::unordered_map<std::string, Obj*>;

        void generate_graph(const Ast* pnode, EvalCtx& ctx, Graph& graph);
    }
}

#endif
