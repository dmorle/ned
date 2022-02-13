#ifndef NED_INTERP_H
#define NED_INTERP_H

#include <ned/core/graph.h>
#include <ned/lang/obj.h>

#include <array>
#include <vector>
#include <tuple>

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
                using Connection = struct { uint64_t node; std::string name; };

                core::EdgeInfo info;
                std::map<std::string, Connection> md_inps;
                std::map<std::string, std::vector<Connection>> md_outs;
            };

            struct TensorBuilder
            {
                uint64_t fwd_edge;
                uint64_t bwd_edge;
                uint64_t init;
            };

            struct NodeBuilder
            {
                std::string name;
                std::map<std::string, uint64_t> inps;
                std::map<std::string, uint64_t> outs;
                std::map<std::string, std::unique_ptr<core::Config>> configs;
            };

            struct InitBuilder
            {
                std::string name;
                std::map<std::string, std::unique_ptr<core::Config>> configs;
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
                std::map<std::string, std::unique_ptr<core::Config>> configs;
            };

            uint64_t root = 0;
            bool is_exported = false;

            // 0 maps to null for all the deep learning objects
            std::vector<EdgeBuilder*>   edges   = { nullptr };
            std::vector<NodeBuilder*>   nodes   = { nullptr };
            std::vector<InitBuilder*>   inits   = { nullptr };
            std::vector<TensorBuilder*> tensors = { nullptr };
            std::vector<BlockBuilder*>  blocks  = { nullptr };

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

            bool add_ndcfg(const std::string& name, uint64_t node, core::Config* pconfig);
            bool add_bkcfg(const std::string& name, uint64_t block, core::Config* pconfig);
            bool add_incfg(const std::string& name, uint64_t init, core::Config* pconfig);

            bool set_ndinp(const std::string& name, uint64_t node, uint64_t edge);
            bool set_ndout(const std::string& name, uint64_t node, uint64_t edge);

            bool set_bkprt(uint64_t block, uint64_t parent_block);
            bool set_bkinp(const std::string& name, uint64_t block, uint64_t tensor);
            bool set_bkout(const std::string& name, uint64_t block, uint64_t tensor);
            bool set_bkext(const std::string& name, uint64_t block, uint64_t tensor);
            bool set_bkexp(const std::string& name, uint64_t block, uint64_t tensor);

            bool export_graph(core::Graph& graph);
        };

        class CallStack
        {
            std::array<Obj, (size_t)1e6> stack;
            size_t sp;

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
