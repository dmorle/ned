#ifndef NED_INTERP_H
#define NED_INTERP_H

#include <ned/core/graph.h>
#include <ned/lang/obj.h>

#include <array>

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
            std::vector<core::Edge*> edge_buffer;
            std::vector<core::Node*> node_buffer;
            std::vector<core::Block*> block_buffer;

            std::map<std::string, core::IOEdge> exports;

        public:
            GraphBuilder() {}

            bool create_edge(core::Edge* pedge);
            bool create_node(const std::string& name, core::Node* pnode);
            bool create_block(const std::string& name, core::Block* pblock);
            
            static bool set_child(core::Block* pparent, core::Block* pchild);
            static bool set_ndinp(core::Node* pnode, core::Edge* pedge, const std::string& name);
            static bool set_ndout(core::Node* pnode, core::Edge* pedge, const std::string& name);
            static bool set_bkinp(core::Block* pblock, core::Edge* pforward, core::Edge* pbackward, const std::string& name);
            static bool set_bkout(core::Block* pblock, core::Edge* pforward, core::Edge* pbackward, const std::string& name);

            static bool set_weight(core::Block* pblock, core::Edge* pforward, core::Edge* pbackward, const std::string& name);
            static bool set_export(core::Edge* pforward, core::Edge* pbackward, const std::string& name);

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

        bool exec(CallStack& stack, ProgramHeap& heap, ByteCode& byte_code, std::string entry_point, core::Graph& graph);
    }
}

#endif
