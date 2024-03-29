#ifndef NED_INTERP_H
#define NED_INTERP_H

#include <ned/core/graph.h>
#include <ned/lang/obj.h>

#include <array>
#include <vector>
#include <tuple>
#include <variant>

namespace nn
{
    namespace lang
    {
        using CodeSegPtr = uint8_t*;
        using DataSegPtr = Obj*;
        using ProcOffsets = std::map<std::string, size_t>;
        struct ByteCode;

        class GraphBuilder
        {
            struct EdgeBuilder
            {
                struct Connector { uint64_t node = 0; std::string name; };

                core::EdgeInfo info;
                std::map<std::string, Connector> md_inps;
                std::map<std::string, std::vector<Connector>> md_outs;

                bool is_merged = false;
                uint64_t tgt_edge = 0;
                
                // feilds used for exporting
                core::Edge* edge;
                struct TensorEdge { uint64_t tensor; bool fwd_edge; };
                std::vector<TensorEdge> tensors;  // Its possible for multiple tensors to reference a single edge
            };

            struct TensorBuilder
            {
                uint64_t fwd_edge = 0;
                uint64_t bwd_edge = 0;
                uint64_t init = 0;

                // fields used for exporting
                union
                {
                    core::Tensor tensor;
                    core::Parameter param = { nullptr, nullptr, nullptr, nullptr };
                };
                
                enum
                {
                    NON_EXPORTED = 0,
                    TENSOR,
                    PARAMETER
                } ty = NON_EXPORTED;

                TensorBuilder() {}
                ~TensorBuilder() {}
            };

            struct NodeBuilder
            {
                std::string name;
                std::vector<std::string> inp_order;
                std::vector<std::string> out_order;
                std::map<std::string, std::vector<uint64_t>> inps;
                std::map<std::string, uint64_t> outs;
                std::map<std::string, core::Config> configs;
                uint64_t parent = 0;
                
                // used only for exporting
                core::Node* node;
            };

            struct InitBuilder
            {
                std::string name;
                std::map<std::string, core::Config> configs;
            };

            struct BlockBuilder
            {
                std::string name;
                std::map<std::string, uint64_t> inps;
                std::map<std::string, uint64_t> outs;
                std::map<std::string, uint64_t> exts;  // Local to the block
                std::map<std::string, uint64_t> exps;  // Local to the block
                uint64_t parent = 0;  // 0 if it is the root block or it is uninitialized
                std::map<std::string, uint64_t> sub_blocks;
                std::map<std::string, uint64_t> sub_nodes;
                std::map<std::string, core::Config> configs;
                
                // used only for exporting
                core::Block* block;
            };

            uint64_t root = 0;
            bool started_export = false;
            bool is_exported = false;

            // 0 maps to null for all the deep learning objects
            std::vector<EdgeBuilder*>   edges   = { nullptr };
            std::vector<NodeBuilder*>   nodes   = { nullptr };
            std::vector<InitBuilder*>   inits   = { nullptr };
            std::vector<TensorBuilder*> tensors = { nullptr };
            std::vector<BlockBuilder*>  blocks  = { nullptr };

            uint64_t edge_lookup(uint64_t edge);
            inline bool edge_exists(uint64_t edge);
            inline bool node_exists(uint64_t node);
            inline bool init_exists(uint64_t init);
            inline bool tensor_exists(uint64_t tensor);
            inline bool block_exists(uint64_t block);

            static std::string current_mode();

        public:
            GraphBuilder();
            GraphBuilder(const GraphBuilder&) = delete;
            GraphBuilder(GraphBuilder&&) = delete;
            GraphBuilder& operator=(const GraphBuilder&) = delete;
            GraphBuilder& operator=(GraphBuilder&&) = delete;
            ~GraphBuilder();
            
            bool create_edg(Obj& obj, const core::EdgeInfo& info);
            bool create_tsr(Obj& obj);
            bool create_nde(Obj& obj, const std::string& name);
            bool create_ini(Obj& obj, const std::string& name);
            bool create_blk(Obj& obj, const std::string& name);

            bool get_fwd(Obj& obj, uint64_t tensor);
            bool get_bwd(Obj& obj, uint64_t tensor);
            bool get_ini(Obj& obj, uint64_t tensor);
            bool set_fwd(uint64_t tensor, uint64_t edge);
            bool set_bwd(uint64_t tensor, uint64_t edge);
            bool set_ini(uint64_t tensor, uint64_t init);
            bool mrg(uint64_t lhs_edge, uint64_t rhs_edge);

            bool get_tshp(Obj& obj, ProgramHeap& heap, uint64_t tensor);
            bool get_tfty(Obj& obj, ProgramHeap& heap, uint64_t tensor);
            bool get_eshp(Obj& obj, ProgramHeap& heap, uint64_t edge);
            bool get_efty(Obj& obj, ProgramHeap& heap, uint64_t edge);
            bool get_einp(Obj& obj, ProgramHeap& heap, uint64_t edge);

            bool add_ndcfg(const std::string& name, uint64_t node, const core::Config& cfg);
            bool add_bkcfg(const std::string& name, uint64_t block, const core::Config& cfg);
            bool add_incfg(const std::string& name, uint64_t init, const core::Config& cfg);

            bool set_ndprt(uint64_t node, uint64_t parent_block);
            bool set_ndinp(const std::string& name, uint64_t node, uint64_t edge);
            bool set_ndout(const std::string& name, uint64_t node, uint64_t edge);

            bool set_bkprt(uint64_t block, uint64_t parent_block);
            bool set_bkinp(const std::string& name, uint64_t block, uint64_t tensor);
            bool set_bkout(const std::string& name, uint64_t block, uint64_t tensor);
            bool set_bkext(const std::string& name, uint64_t block, uint64_t tensor);
            bool set_bkexp(const std::string& name, uint64_t block, uint64_t tensor);

            bool export_graph(core::Graph& graph);

        private:
            bool export_edge(EdgeBuilder* edge);
            bool export_node(NodeBuilder* node);
            bool bind_edge(EdgeBuilder* edge);
            bool export_init(core::Init*& init, uint64_t i);
            bool export_tensor(uint64_t i);
            bool export_block(core::Block& block, uint64_t i);
            bool bind_block(core::Block& block, uint64_t i);
            bool export_io(core::Graph& graph, const std::string& prefix, uint64_t block);
        };

        class CallStack
        {
            std::array<Obj, (size_t)1e6> stack;
            size_t sp = 0;

        public:
            bool pop(Obj& obj);
            bool del(size_t i);
            bool get(size_t i, Obj& obj);
            bool push(Obj obj);
        };

        bool exec(CallStack& stack, ProgramHeap& heap, GraphBuilder& builder, ByteCode& byte_code, std::string entry_point);
    }
}

#endif
