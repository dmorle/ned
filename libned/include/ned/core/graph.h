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
            struct Connector { Node* node = nullptr; std::string name; };

            EdgeInfo info;
            std::map<std::string, Connector> md_inps;
            std::map<std::string, std::vector<Connector>> md_outs;
            mutable void* opaque = nullptr;
        };

        struct Block;
        struct Node
        {
            std::string name;
            std::map<std::string, std::unique_ptr<Config>> configs;
            std::map<std::string, Edge*> inps;
            std::map<std::string, Edge*> outs;
            Block* parent = nullptr;
            mutable void* opaque = nullptr;
        };

        struct Tensor
        {
            Edge* forward;
            Edge* backward;
        };

        struct Parameter
        {
            Edge* forward  = nullptr;
            Edge* backward = nullptr;
            void* data     = nullptr;
            Init* init     = nullptr;  // This has to be a pointer because Parameter needs to have a copy-constructor
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
            std::map<std::string, Node*> sub_nodes;
        };

        struct Graph
        {
            Block model;
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
