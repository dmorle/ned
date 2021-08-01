#ifndef NN_EVAL_H
#define NN_EVAL_H

#include <libnn/core/graph.h>

#include <string>
#include <map>

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

        class Scope
        {
            std::map<std::string, Obj*> data;
            
        public:
            Scope();
            ~Scope();

            Obj* operator[](const std::string& idn);
        };

        enum class EvalState
        {
            PACKAGE,
            MODULE,
            DEF,
            INTR,
            FN
        };

        class EvalCtx
        {
            friend class AstDef;
            friend class AstFn;
            friend class AstIntr;
            friend class AstModule;
            friend class AstModImp;

            std::map<std::string, Obj*> defs;
            std::map<std::string, Obj*> fns;
            std::map<std::string, Obj*> intrs;
            std::map<std::string, Obj*> mods;
            std::map<std::string, Obj*> packs;

        public:
            EvalCtx();

            Obj* operator[](const std::string& idn);

            bool contains(const std::string& name);
            void insert(std::pair<std::string, Obj*> val);
            
            Graph* pgraph;
            Scope* pscope;

            EvalState state;
        };
    }
}

#endif
