#ifndef NED_GRAPH_H
#define NED_GRAPH_H

#include <map>
#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include <ned/core/tensor.h>

namespace nn
{
    namespace lang { class Obj; }

    namespace core
    {
        class InvalidGraph :
            public std::exception
        {
            InvalidGraph(const std::string& errmsg);
        };

        struct Node;

        struct Edge
        {
            tensor_dsc dsc = tensor_dsc{};
            Node* input = nullptr;
            int inpid = -1;
            std::vector<std::pair<Node*, int>> outputs = {};
            mutable void* opaque = nullptr;
        };

        struct Node
        {
            std::string name;
            std::vector<std::shared_ptr<lang::Obj>> cargs;
            std::vector<Edge*> inputs;
            std::vector<Edge*> outputs;
            mutable void* opaque = nullptr;
        };

        struct Graph
        {
            std::map<std::string, Edge*> inputs;
            std::vector<Edge*> outputs;
        };

        // modifies the given graph to include a gradient. Output gradient edges are returned
        std::map<std::string, Edge*> generate_grad(Graph* pgraph);
        void unroll_graph(Graph* pgraph, std::map<std::string, int> inp_counts, Graph& result);
    }
}

#endif
