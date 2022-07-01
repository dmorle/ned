#include <ned/errors.h>
#include <ned/core/reduce.h>

namespace nn
{
    namespace core
    {
        EdgeView identity_edge_view(const MdEdge& edge)
        {
            // TODO: figure out edge views
            return EdgeView();
        }

        bool MdGraph::init(Graph& graph)
        {
            // No need to keep track of the backward edges at this point.  Either they've been bound
            // to the forward edges by an optimizer, or its for inference mode and they don't matter

            // translating the model outputs
            for (const auto& [name, tensor] : graph.outs)
            {
                outs[name] = make_edge();
                tensor.forward->opaque = as_ptr(outs.at(name));
                if (init_edge(*tensor.forward))
                    return true;
            }
            // translating the model exports
            for (const auto& [name, tensor] : graph.exports)
            {
                exps[name] = make_edge();
                tensor.forward->opaque = as_ptr(outs.at(name));
                if (init_edge(*tensor.forward))
                    return true;
            }
            // binding the model outputs
            for (const auto& [name, tensor] : graph.outs)
                if (bind_edge(*tensor.forward))
                    return true;
            // binding the model exports
            for (const auto& [name, tensor] : graph.exports)
                if (bind_edge(*tensor.forward))
                    return true;
            
            // At this point in the graph representation, the focus is on execution rather than representation.
            // As a result, if any inputs or model weights are not required to produce the model's outputs,
            // those portions of the model will not get translated.
            // Also from this translation, this will automatically provide graph trimming for free since
            // any nodes/edges in the graph which aren't required to compute the outputs won't show up in the dfs.
            // Note that for model training, optimizers will connect the bwd edges of weights to their fwd edges
            // This will cause the forward output edges to be dependant on the backwards edges

            for (const auto& [name, tensor] : graph.inps)
                if (tensor.forward->opaque)
                {
                    // As mentioned above: only using the inputs that showed up in the dfs.
                    inps[name] = edge_ptr(tensor.forward->opaque);
                }
            
            for (const auto& [name, parameter] : graph.weights)
                if (parameter.forward->opaque)
                {
                    // For model weights, also initialize the data field at this point
                    exts[name] = edge_ptr(parameter.forward->opaque);
                    get(exts.at(name)).data = parameter.data;
                }
            return false;
        }

        MdEdgeRef MdGraph::make_edge()
        {
            MdEdgeRef ret{ edges.size() };
            edges.push_back(MdEdge());
            return ret;
        }

        MdNodeRef MdGraph::make_node()
        {
            MdNodeRef ret{ nodes.size() };
            nodes.push_back(MdNode());
            return ret;
        }

        bool MdGraph::init_edge(const Edge& edge)
        {
            MdEdgeRef edge_ref = edge_ptr(edge.opaque);
            get(edge_ref).fp = edge.info.fty;
            get(edge_ref).shape = edge.info.dims;

            for (const auto& [mode, conn] : edge.md_inps)
                if (check_mode(mode))
                {
                    if (get(edge_ref).inp)
                        return error::graph("Unable to resolve edge input for given execution mode");
                    
                    if (conn.node->opaque)
                    {
                        // Already initialized, just attach it to the edge and continue checking
                        get(edge_ref).inp = node_ptr(conn.node->opaque);
                    }
                    else
                    {
                        // Initialize the node, then dfs on it
                        get(edge_ref).inp = make_node();
                        conn.node->opaque = as_ptr(get(edge_ref).inp);
                        if (init_node(*conn.node))
                            return true;
                    }
                }
            return false;
        }

        bool MdGraph::init_node(const Node& node)
        {
            MdNodeRef node_ref = node_ptr(node.opaque);
            get(node_ref).name = node.name;
            get(node_ref).configs = node.configs;

            for (const auto& name : node.inp_order)
            {
                // Making sure that the edge agrees on the node being part of the given execution mode
                for (const auto& edge : node.inps.at(name))
                {
                    for (const auto& [mode, conns] : edge->md_outs)
                        for (const auto& conn : conns)
                            if (conn.node == &node)
                            {
                                if (!check_mode(mode))
                                    return error::graph("Conflicting node requirements");
                                goto all_good;
                            }
                    return error::graph("Connection mismatch between node and edge");

                all_good:
                    if (edge->opaque)
                    {
                        // Already initialized, just attach it to the node and move on to the next edge
                        MdEdgeRef md_edge = edge_ptr(edge->opaque);
                        get(node_ref).inps.push_back({ md_edge, identity_edge_view(get(md_edge)) });
                    }
                    else
                    {
                        // Initialize it, then dfs on it
                        MdEdgeRef md_edge = make_edge();
                        get(node_ref).inps.push_back({ md_edge, identity_edge_view(get(md_edge)) });
                        edge->opaque = as_ptr(std::get<0>(get(node_ref).inps.back()));
                        if (init_edge(*edge))
                            return true;
                    }
                }
            }
            return false;
        }

        bool MdGraph::bind_edge(const Edge& edge)
        {
            // All the mode checks should be done at this point, so its just
            // connecting the edge and node outputs for anything that got initialized
            MdEdgeRef edge_ref = edge_ptr(edge.opaque);

            // Checking if the edge was already bound (in case of loops in the graph)
            if (get(edge_ref).outs.size())
                return false;

            // It wasn't already bound. So do the binding, then dfs
            for (const auto& [mode, conns] : edge.md_outs)
                for (const auto& conn : conns)
                    if (conn.node->opaque)
                    {
                        // Only connecting to nodes that weren't pruned during the init
                        // The outputs are unordered, so it this shouldn't be an issue
                        get(edge_ref).outs.push_back(node_ptr(conn.node->opaque));
                    }
            for (const auto& [mode, conn] : edge.md_inps)
                if (conn.node->opaque)
                {
                    if (bind_node(*conn.node))
                        return true;
                }
            return false;
        }

        bool MdGraph::bind_node(const Node& node)
        {
            MdNodeRef node_ref = node_ptr(node.opaque);

            // Checking if the node was already bound
            if (get(node_ref).outs.size())
                return false;

            // It wasn't already bound. So do the binding, the dfs on the edges
            for (const auto& [name, edge] : node.outs)
                if (edge->opaque)
                {
                    // Similar to the edges, the node outputs are unordered
                    get(node_ref).outs.push_back(edge_ptr(edge->opaque));
                }
            for (const auto& name : node.inp_order)
                for (const auto& edge : node.inps.at(name))
                    if (edge->opaque)
                    {
                        if (bind_edge(*edge))
                            return true;
                    }
            return false;
        }

        bool MdGraph::check_mode(const std::string& mode)
        {
            const char* m_md = exec_md.c_str();
            const char* o_md = mode.c_str();
            while (*o_md)
                // This will take care of the condition where m_md ran out of characters but o_md kept going
                if (*o_md != *m_md)
                    return false;
                
            // if m_md is more specific then its a match
            return *m_md == '\0' || *m_md == '.';
        }
    
        MdEdge& MdGraph::get(MdEdgeRef ref) noexcept { return edges[ref.val]; }
        const MdEdge& MdGraph::get(MdEdgeRef ref) const noexcept { return edges[ref.val]; }
        MdNode& MdGraph::get(MdNodeRef ref) noexcept { return nodes[ref.val]; }
        const MdNode& MdGraph::get(MdNodeRef ref) const noexcept { return nodes[ref.val]; }
        void* MdGraph::as_ptr(MdEdgeRef ref) const noexcept { return reinterpret_cast<void*>(ref.val); }
        void* MdGraph::as_ptr(MdNodeRef ref) const noexcept { return reinterpret_cast<void*>(ref.val); }
        MdEdgeRef MdGraph::edge_ptr(void* ptr) const noexcept { return { reinterpret_cast<size_t>(ptr) }; }
        MdNodeRef MdGraph::node_ptr(void* ptr) const noexcept { return { reinterpret_cast<size_t>(ptr) }; }
    }
}