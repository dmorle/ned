#ifndef NED_GRAPH_H
#define NED_GRAPH_H

#include <ned/core/config.h>
#include <ned/core/setup.h>

#include <map>
#include <tuple>
#include <vector>
#include <string>

namespace nn
{
    namespace lang { union Obj; }

    namespace core
    {
        struct Node;
        struct Edge
        {
            using Connector = struct { Node* node; std::string name; };

            EdgeInfo info;
            std::map<std::string, Connector> md_inps;
            std::map<std::string, std::vector<Connector>> md_outs;
            mutable void* opaque = nullptr;
        };

        struct Node
        {
            std::string name;
            std::map<std::string, std::unique_ptr<Config>> configs;
            std::map<std::string, Edge*> inps;
            std::map<std::string, Edge*> outs;
            mutable void* opaque = nullptr;
        };

        struct Tensor
        {
            Edge* forward;
            Edge* backward;
        };

        struct Parameter
        {
            Edge* forward;
            Edge* backward;
            Init* init;
            void* data = nullptr;
        };

        struct Block
        {
            std::string name;
            std::map<std::string, std::unique_ptr<Config>> configs;
            std::map<std::string, Tensor> inps;
            std::map<std::string, Tensor> outs;
            std::map<std::string, Tensor> exports;  // Local to the block
            std::map<std::string, Parameter> weights;  // Local to the block
            Block* parent = nullptr;  // null if its the root block or uninitialized
            std::map<std::string, Block*> sub_blocks;
        };

        struct Graph
        {
            Block* model;
            std::map<std::string, Tensor> inps;
            std::map<std::string, Tensor> outs;
            std::map<std::string, Tensor> exports;  // Global across the graph
            std::map<std::string, Parameter> weights;  // Global across the graph
            std::vector<std::string> eval_modes;
        };

        void initialize_weights(Graph& graph);
    }
}

#endif
