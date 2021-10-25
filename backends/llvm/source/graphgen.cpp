#include <nedvm/graphgen.h>

namespace nn
{
    namespace nedvm
    {
        GraphCompiler::GraphCompiler(const core::Graph* pgraph) :
            ctx{},
            builder{ llvm::IRBuilder<>(ctx) },
            mod{ "main", ctx }
        {
            this->pgraph = pgraph;
            for (const core::Edge* pedge : pgraph->outputs)
                initEdgeOpaque(pedge);
        }

        GraphCompiler::~GraphCompiler()
        {
            for (const core::Edge* pedge : pgraph->outputs)
                delEdgeOpaque(pedge);
        }

        void GraphCompiler::initEdgeOpaque(const core::Edge* pedge)
        {
            if (pedge->opaque)
                return;

            for (const core::Edge* pedge : pedge->input->inputs)
                initEdgeOpaque(pedge);
            
            pedge->opaque = new EdgeData();
        }

        void GraphCompiler::delEdgeOpaque(const core::Edge* pedge)
        {
            if (!pedge->opaque)
                return;

            for (const core::Edge* pedge : pedge->input->inputs)
                delEdgeOpaque(pedge);

            delete (EdgeData*)pedge->opaque;
            pedge->opaque = nullptr;
        }
    }
}
