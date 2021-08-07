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

        using Scope = std::map<std::string, std::unique_ptr<Obj>>;

        class EvalCtx
        {
            friend class AstDef;
            friend class AstFn;
            friend class AstIntr;
            friend class AstModule;
            friend class AstModImp;

            std::map<std::string, std::unique_ptr<Obj>> defs;
            std::map<std::string, std::unique_ptr<Obj>> fns;
            std::map<std::string, std::unique_ptr<Obj>> intrs;
            std::map<std::string, std::unique_ptr<Obj>> mods;
            std::map<std::string, std::unique_ptr<Obj>> packs;

            std::vector<std::string> model_params;
            Graph* pgraph;
            Scope* pscope;

        public:
            EvalCtx();

            std::unique_ptr<Obj> operator[](const std::string& idn);

            bool contains(const std::string& name);
            void insert(std::pair<std::string, std::unique_ptr<Obj>>& val);

            Graph& graph() noexcept;
            Scope& scope() noexcept;

            Scope* swap_scope(Scope* npscope);

            // identify an edge as a model parameter
            void add_param(const std::string& param_name);

            EvalState state;
        };
    }
}

#endif
