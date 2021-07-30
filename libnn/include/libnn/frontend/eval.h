#ifndef NN_EVAL_H
#define NN_EVAL_H

#include <string>
#include <unordered_map>

namespace nn
{
    namespace impl
    {
        class Obj;
        class AstBlock;

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
            std::unordered_map<std::string, Obj*> data;
            
        public:
            Scope();

            Obj* operator[](const std::string& idn);

            void release();
        };

        class EvalCtx
        {
            std::vector<Scope> scope_stack;

        public:
            EvalCtx();

            Obj* operator[](std::string& idn);
        };
    }
}

#endif

