#ifndef NED_CORE_GRAPH_H
#define NED_CORE_GRAPH_H

#include <ned/core/config.h>

#include <map>
#include <tuple>
#include <vector>
#include <string>
#include <functional>

namespace nn
{
    namespace core
    {
        struct Init
        {
            std::string name;
            std::map<std::string, Config> configs;
        };

        struct InitImpl
        {
            std::map<std::string, Config> signature;
            std::function<void* (EdgeFty, const Init&, const EdgeInfo&)> init;
        };
        inline std::map<std::string, InitImpl> inits = {};

        struct Node;
        struct Edge
        {
            struct InpConnector { Node* node = nullptr; std::string name; };                  // node to edge
            struct OutConnector { Node* node = nullptr; std::string name; size_t idx = 0; };  // edge to node
            
            // InpConnector doesn't need idx cause currently ned doesn't support variadic node outputs.
            // During mode collpase, the idx can be determined from the output order of the feeding node.

            EdgeInfo info;
            std::map<std::string, InpConnector> md_inps;
            std::map<std::string, std::vector<OutConnector>> md_outs;
            mutable void* opaque = nullptr;
        };

        struct Block;
        struct Node
        {
            std::string name;
            std::map<std::string, Config> configs;
            // During mode collapse, node inputs and outputs are purely positional,
            // so the node needs to keep track of the order of the keyword inputs/outputs
            std::vector<std::string> inp_order;
            std::vector<std::string> out_order;
            std::unordered_map<std::string, std::vector<Edge*>> inps;
            std::unordered_map<std::string, Edge*> outs;
            Block* parent = nullptr;
            mutable void* opaque = nullptr;
        };

        struct Tensor
        {
            Edge* forward  = nullptr;
            Edge* backward = nullptr;
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
            std::map<std::string, Config> configs;
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

        struct InpRef
        {
            enum class Type
            {
                INPUT,
                WEIGHT
            } ty;
            std::string name;

            auto operator<=>(const InpRef&) const = default;
        };

        struct OutRef
        {
            enum class Type
            {
                OUTPUT,
                EXPORT
            } ty;
            std::string name;

            auto operator<=>(const OutRef&) const = default;
        };

        struct GraphMod
        {
            struct IOMap
            {
                std::map<InpRef, OutRef> inp_map;  // Mapping of mod outputs to graph inputs
                std::map<OutRef, InpRef> out_map;  // Mapping of graph outputs to mod inputs
            };

            IOMap io_map;
            Graph graph;
        };

        bool copy_graph(Graph& out, const Graph& graph);

        bool init_weights(Graph& graph);
        bool save_graph(const Graph& graph, std::vector<uint8_t>& out);
        bool load_graph(Graph& graph, const std::vector<uint8_t>& inp);

        // mods gets destroyed, and its contents get added to graph
        bool attach_graph(Graph& graph, const std::string& name, std::vector<GraphMod>& mods);
    }
}

#endif
