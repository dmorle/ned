#include <ned/core/graph.h>
#include <ned/errors.h>

#include <set>

namespace nn
{
    namespace core
    {
        bool attach_graph(Graph& graph, const std::string& name, std::vector<GraphMod>& mods)
        {
            // Attaching the graphs at the edge/node level
            for (GraphMod& mod : mods)
            {
                // Attaching the mod outputs to the graph inputs
                for (const auto& [gref, mref] : mod.io_map.inp_map)
                {
                    Tensor gten;
                    switch (gref.ty)
                    {
                    case InpRef::Type::INPUT:
                        if (!graph.inps.contains(gref.name))
                            return error::graph("Found reference to non-existant graph input %", gref.name);
                        gten = graph.inps.at(gref.name);
                        graph.inps.erase(gref.name);  // removing the input from graph
                        break;
                    case InpRef::Type::WEIGHT:
                        if (!graph.weights.contains(gref.name))
                            return error::graph("Found reference to non-existant graph weight %", gref.name);
                        // Even though it has the data and init fields, casting it will be fine.
                        // This is cause the gten will be kept, and mten will be deleted for inp_map
                        gten = *(Tensor*)&graph.weights.at(gref.name);
                        break;
                    default:
                        return error::graph("Internal error: enum out of range");
                    }

                    Tensor mten;
                    switch (mref.ty)
                    {
                    case OutRef::Type::OUTPUT:
                        if (!mod.graph.outs.contains(mref.name))
                            return error::graph("Found reference to non-existant mod output %", mref.name);
                        mten = mod.graph.outs.at(mref.name);
                        mod.graph.outs.erase(mref.name);  // removing the output from mod
                        break;
                    case OutRef::Type::EXPORT:
                        if (!mod.graph.exports.contains(mref.name))
                            return error::graph("Found reference to non-existant mod weight %", mref.name);
                        mten = mod.graph.outs.at(mref.name);
                        break;
                    default:
                        return error::graph("Internal error: enum out of range");
                    }

                    // Checking the float width and shape
                    if (mten.forward->info != gten.forward->info)
                        return error::graph("Forward edge mismatch found during graph attachment");
                    if (mten.backward->info != gten.backward->info)
                        return error::graph("Backward edge mismatch found during graph attachment");

                    // Checking for any conflicts in execution mode
                    for (const auto& [exec_md, _] : mten.forward->md_inps)
                        if (gten.forward->md_inps.contains(exec_md))
                            return error::graph("Found conflicting forward execution modes between % and %", gref.name, mref.name);
                    for (const auto& [exec_md, _] : gten.backward->md_inps)
                        if (mten.backward->md_inps.contains(exec_md))
                            return error::graph("Found conflicting backward execution modes between % and %", gref.name, mref.name);

                    // Everything should be good to attach (mod output to graph input)
                    
                    // forward edge outputs
                    for (const auto& [exec_md, conns] : mten.forward->md_outs)
                        for (const auto& conn : conns)
                        {
                            conn.node->inps.at(conn.name)[conn.idx] = gten.forward;  // node to edge
                            gten.forward->md_outs[exec_md].push_back(conn);          // edge to node
                        }
                    // forward edge inputs
                    for (const auto& [exec_md, conn] : mten.forward->md_inps)
                    {
                        conn.node->outs.at(conn.name) = gten.forward;
                        gten.forward->md_inps[exec_md] = conn;
                    }
                    delete mten.forward;  // The edge isn't reference by the graph anymore
                    // backward edge outputs
                    for (const auto& [exec_md, conns] : mten.backward->md_outs)
                        for (const auto& conn : conns)
                        {
                            conn.node->inps.at(conn.name)[conn.idx] = gten.backward;  // node to edge
                            gten.backward->md_outs[exec_md].push_back(conn);
                        }
                    // backward edge inputs
                    for (const auto& [exec_md, conn] : mten.backward->md_inps)
                    {
                        conn.node->outs.at(conn.name) = gten.backward;
                        gten.backward->md_inps[exec_md] = conn;
                    }
                    delete mten.backward;  // The edge isn't reference by the graph anymore
                }

                // Attaching the graph outputs to the mod inputs
                for (const auto& [gref, mref] : mod.io_map.out_map)
                {
                    Tensor gten;
                    switch (gref.ty)
                    {
                    case OutRef::Type::OUTPUT:
                        if (!graph.outs.contains(gref.name))
                            return error::graph("Found reference to non-existant graph input %", gref.name);
                        gten = graph.outs.at(gref.name);
                        graph.outs.erase(gref.name);  // removing the output from graph
                        break;
                    case OutRef::Type::EXPORT:
                        if (!graph.exports.contains(gref.name))
                            return error::graph("Found reference to non-existant graph export %", gref.name);
                        gten = graph.exports.at(gref.name);
                        break;
                    default:
                        return error::graph("Internal error: enum out of range");
                    }

                    Tensor mten;
                    switch (mref.ty)
                    {
                    case InpRef::Type::INPUT:
                        if (!mod.graph.inps.contains(mref.name))
                            return error::graph("Found reference to non-existant mod input %", mref.name);
                        mten = mod.graph.inps.at(mref.name);
                        mod.graph.inps.erase(mref.name);  // removing the input from mod
                        break;
                    case InpRef::Type::WEIGHT:
                        if (!mod.graph.weights.contains(gref.name))
                            return error::graph("Found reference to non-existant mod weight %", mref.name);
                        // Even though it has the data and init fields, casting it will be fine.
                        // This is cause the mten will be kept, and gten will be deleted for out_map
                        mten = *(Tensor*)&mod.graph.weights.at(mref.name);
                        break;
                    default:
                        return error::graph("Internal error: enum out of range");
                    }

                    // Checking for any conflicts in execution mode
                    for (const auto& [exec_md, _] : gten.forward->md_inps)
                        if (mten.forward->md_inps.contains(exec_md))
                            return error::graph("Found conflicting forward execution modes between % and %", gref.name, mref.name);
                    for (const auto& [exec_md, _] : mten.backward->md_inps)
                        if (gten.backward->md_inps.contains(exec_md))
                            return error::graph("Found conflicting backward execution modes between % and %", gref.name, mref.name);

                    // Everything should be good to attach (mod output to graph input)

                    // forward edge outputs
                    for (const auto& [exec_md, conns] : gten.forward->md_outs)
                        for (const auto& conn : conns)
                        {
                            conn.node->inps.at(conn.name)[conn.idx] = mten.forward;  // node to edge
                            mten.forward->md_outs[exec_md].push_back(conn);          // edge to node
                        }
                    // forward edge inputs
                    for (const auto& [exec_md, conn] : gten.forward->md_inps)
                    {
                        conn.node->outs.at(conn.name) = mten.forward;
                        mten.forward->md_inps[exec_md] = conn;
                    }
                    delete gten.forward;  // The edge isn't reference by the graph anymore
                    // backward edge outputs
                    for (const auto& [exec_md, conns] : gten.backward->md_outs)
                        for (const auto& conn : conns)
                        {
                            conn.node->inps.at(conn.name)[conn.idx] = mten.backward;  // node to edge
                            mten.backward->md_outs[exec_md].push_back(conn);
                        }
                    // backward edge inputs
                    for (const auto& [exec_md, conn] : gten.backward->md_inps)
                    {
                        conn.node->outs.at(conn.name) = mten.backward;
                        mten.backward->md_inps[exec_md] = conn;
                    }
                    delete gten.backward;  // The edge isn't reference by the graph anymore
                }
            }
            
            // Adding the new inputs/outputs/weights/exports from each of the mods to the graph
            for (const GraphMod& mod : mods)
            {
                for (const auto& [name, tensor] : mod.graph.inps)
                {
                    if (graph.inps.contains(name))
                        return error::graph("Input name % conflicts in graph attachment", name);
                    graph.inps[name] = tensor;
                }
                for (const auto& [name, tensor] : mod.graph.outs)
                {
                    if (graph.outs.contains(name))
                        return error::graph("Output name % conflicts in graph attachment", name);
                    graph.outs[name] = tensor;
                }
                for (const auto& [name, tensor] : mod.graph.exports)
                {
                    if (graph.exports.contains(name))
                        return error::graph("Export name % conflicts in graph attachment", name);
                    graph.exports[name] = tensor;
                }
                for (const auto& [name, weight] : mod.graph.weights)
                {
                    if (graph.weights.contains(name))
                        return error::graph("Weight name % conflicts in graph attachment", name);
                    graph.weights[name] = weight;
                }
            }
            
            // Creating a new top-level block to contain graph and the mods

            Block* old_model = new Block(graph.model);
            graph.model = Block();
            // The top-level block's inputs and outputs will mimic the resulting graph's
            // but during attachment, no extra weights or exports will get created
            for (const auto& [name, tensor] : graph.inps)
                graph.model.inps[name] = tensor;
            for (const auto& [name, tensor] : graph.outs)
                graph.model.outs[name] = tensor;
            
            // Adding the sub-blocks to the new top-level block
            std::unordered_map<std::string, size_t> name_count;
            graph.model.sub_blocks[old_model->name + "~" +
                std::to_string(name_count[old_model->name])] = old_model;
            for (GraphMod& mod : mods)
                graph.model.sub_blocks[mod.graph.model.name + "~" +
                    std::to_string(name_count[mod.graph.model.name]++)] = &mod.graph.model;

            // Removing the nodes and edges from the mods since their memory is now owned by graph
            for (GraphMod& mod : mods)
            {
                mod.graph.inps.clear();
                mod.graph.outs.clear();
                mod.graph.exports.clear();
                mod.graph.weights.clear();
            }

            return false;
        }
    }
}
