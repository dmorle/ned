#ifndef NED_INTERP_H
#define NED_INTERP_H

#include <ned/core/graph.h>
#include <ned/lang/errors.h>
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

            bool create_edge(RuntimeErrors& errs, core::Edge* pedge);
            bool create_node(RuntimeErrors& errs, const std::string& name, core::Node* pnode);
            bool create_block(RuntimeErrors& errs, const std::string& name, core::Block* pblock);
            
            static bool set_child(RuntimeErrors& errs, core::Block* pparent, core::Block* pchild);
            static bool set_ndinp(RuntimeErrors& errs, core::Node* pnode, core::Edge* pedge, const std::string& name);
            static bool set_ndout(RuntimeErrors& errs, core::Node* pnode, core::Edge* pedge, const std::string& name);
            static bool set_bkinp(RuntimeErrors& errs, core::Block* pblock, core::Edge* pforward, core::Edge* pbackward, const std::string& name);
            static bool set_bkout(RuntimeErrors& errs, core::Block* pblock, core::Edge* pforward, core::Edge* pbackward, const std::string& name);

            static bool set_weight(RuntimeErrors& errs, core::Block* pblock, core::Edge* pforward, core::Edge* pbackward, const std::string& name);
            static bool set_export(RuntimeErrors& errs, core::Edge* pforward, core::Edge* pbackward, const std::string& name);

            bool export_graph(RuntimeErrors& errs, core::Graph& graph);
        };

        class CallStack
        {
            std::array<Obj, (size_t)1e6> stack;
            size_t sp;

        public:
            bool pop(RuntimeErrors& errs, Obj& obj);
            bool del(RuntimeErrors& errs, size_t i);
            bool get(RuntimeErrors& errs, size_t i, Obj& obj);
            bool push(RuntimeErrors& errs, Obj obj);
        };

        bool exec(Errors& errs, CallStack& stack, ProgramHeap& heap, ByteCode& byte_code, std::string entry_point, core::Graph& graph);
    }
}

#endif
