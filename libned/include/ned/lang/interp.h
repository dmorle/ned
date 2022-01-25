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
            std::map<std::string, core::Tensor> exports;
            core::Block* root = nullptr;

        public:
            GraphBuilder();
            
            bool get_forward(Obj& obj, core::Tensor* pten);
            bool get_backward(Obj& obj, core::Tensor* pten);

            bool add_ndcfg(const std::string& name, core::Node* pnode, core::Config* pconfig);
            bool add_bkcfg(const std::string& name, core::Block* pblock, core::Config* pconfig);
            bool add_incfg(const std::string& name, core::Init* pinit, core::Config* pconfig);

            bool set_ndinp(const std::string& name, core::Node* pnode, core::Edge* pedge);
            bool set_ndout(const std::string& name, core::Node* pnode, core::Edge* pedge);

            bool set_bkprt(core::Block* pblock, core::Block* pparent);
            bool set_bkinp(const std::string& name, core::Block* pblock, core::Tensor* pten);
            bool set_bkout(const std::string& name, core::Block* pblock, core::Tensor* pten);

            bool add_extern(const std::string& name, core::Block* pblock, core::Tensor* pten, core::Init* pinit);
            bool add_export(const std::string& name, core::Tensor* pten);

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
