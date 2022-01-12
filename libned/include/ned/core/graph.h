#ifndef NED_GRAPH_H
#define NED_GRAPH_H

#include <ned/core/tensor.h>

#include <map>
#include <tuple>
#include <vector>
#include <string>

namespace nn
{
    namespace lang { union Obj; }

    namespace core
    {
        enum class EdgeFty
        {
            F16,
            F32,
            F64
        };

        struct EdgeInfo
        {
            EdgeFty fty;
            std::vector<size_t> dims;
        };

        struct Node;
        struct Edge
        {
            using Connector = struct { Node* node; std::string name; };
            using Info = struct { EdgeFty fty = EdgeFty::F32; std::vector<size_t> dims; };

            Info info;
            std::map<std::string, Connector> md_inps;
            std::map<std::string, std::vector<Connector>> ctx_outs;
            mutable void* opaque = nullptr;
        };

        struct Node
        {
            std::string name;
            std::map<std::string, lang::Obj> cargs;
            std::map<std::string, Edge*> inps;
            std::map<std::string, Edge*> outs;
            mutable void* opaque = nullptr;
        };

        struct IOEdge
        {
            Edge* forward;
            Edge* backward;
        };

        struct Block
        {
            std::string name;
            std::map<std::string, IOEdge> inps;
            std::map<std::string, IOEdge> outs;
            std::map<std::string, IOEdge> weights;  // Local to the block
            std::map<std::string, IOEdge> exports;  // Local to the block
            std::map<std::string, Block*> sub_blocks;
        };

        struct Graph
        {
            Block* model;
            std::map<std::string, IOEdge> inps;
            std::map<std::string, IOEdge> outs;
            std::map<std::string, IOEdge> weights;  // Global across the graph
            std::map<std::string, IOEdge> exports;  // Global across the graph
            std::vector<std::string> eval_modes;
        };

        size_t fty_size(tensor_dty dty);
        bool fty_str(tensor_dty dty, std::string& str);

        bool validate_block(const Block& blk);
    }
}

#endif
