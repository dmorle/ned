#ifndef NED_INTERP_H
#define NED_INTERP_H

#include <ned/core/graph.h>
#include <ned/lang/obj.h>

#include <array>
#include <vector>

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
            std::vector<core::Init*> init_buffer;
            std::vector<core::Block*> block_buffer;

            std::map<std::string, core::IOEdge> exports;
            core::Block* root;

        public:
            GraphBuilder(const std::string& name);
            Obj get_root();

            bool create_edge(Obj& edge);
            bool create_node(Obj& node, const std::string& name);
            bool create_init(Obj& init, const std::string& name, const std::vector<core::Config*> cfgs);
            bool create_block(Obj& block, const std::string& name, core::Block* parent);
            
            static bool set_ndinp(const std::string& name, core::Node* pnode, core::Edge* pedge);
            static bool set_ndout(const std::string& name, core::Node* pnode, core::Edge* pedge);
            static bool set_bkinp(const std::string& name, core::Block* pblock, core::Edge* pforward, core::Edge* pbackward);
            static bool set_bkout(const std::string& name, core::Block* pblock, core::Edge* pforward, core::Edge* pbackward);

            bool add_extern(const std::string& name, core::Block* pblock, core::Edge* pforward, core::Edge* pbackward, core::Init* pinit);
            bool add_export(const std::string& name, core::Edge* pforward, core::Edge* pbackward);

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
