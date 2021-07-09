#ifndef NN_GENERATOR_H
#define NN_GENERATOR_H

#include <libnn/core/graph.h>
#include <libnn/frontend/parser.h>

namespace nn
{
    namespace frontend
    {
        void generate_graph(const AstNode* pnode, Graph& graph);
    }
}

#endif
