#ifndef NN_EVAL_H
#define NN_EVAL_H

#include <libnn/core/graph.h>

#include <string>
#include <map>
#include <memory>

namespace nn
{
    namespace impl
    {
        class Obj;
        class AstBlock;
        class AstModImp;

        class GenerationError :
            public std::exception
        {
        public:
            std::string errmsg;
            std::vector<std::tuple<std::string, uint32_t, uint32_t>> traceback;

            GenerationError(const std::string& errmsg);

            void tb_touch(AstBlock* pblock);
            void tb_touch(const std::string& file_name, uint32_t line_num, uint32_t col_num);
        };

        enum class EvalState
        {
            CALL,     // evaluating call arguments
            DEFSEQ,   // raw statements in a def
            DEFEXPR,  // part of an expression in a def
            INTR,     // intrinsic
            FN        // function
        };

        using Scope = std::map<std::string, std::shared_ptr<Obj>>;

        class EvalCtx
        {
        public:
            EvalCtx();

            // scope > defs > fns > intrs > mods > packs
            std::shared_ptr<Obj> operator[](const std::string& idn);
            bool contains(const std::string& name);

            Graph& graph() noexcept;
            Scope& scope() noexcept;

            std::map<std::string, std::shared_ptr<Obj>> defs;
            std::map<std::string, std::shared_ptr<Obj>> fns;
            std::map<std::string, std::shared_ptr<Obj>> intrs;
            std::map<std::string, std::shared_ptr<Obj>> mods;
            std::map<std::string, std::shared_ptr<Obj>> packs;

            std::map<std::string, std::shared_ptr<Obj>> statics;

            std::vector<std::string> model_params;
            Graph* pgraph;
            Scope* pscope;

            EvalState state;
            std::string block_name;
        };
    }
}

#endif
