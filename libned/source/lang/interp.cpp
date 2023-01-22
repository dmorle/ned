#include <ned/errors.h>
#include <ned/lang/interp.h>
#include <ned/lang/bytecode.h>

#include <iostream>
#include <vector>
#include <windows.h>

#define oprand (*((size_t*)(code + pc)))

namespace nn
{
    namespace lang
    {
        // state of the interpreter (no, this shouldn't be implemented as a singleton.)
        // programming state
        size_t pc = 0;
        CodeSegPtr code;
        DataSegPtr data;
        bool complete;
        std::vector<size_t> pc_stack;

        // deep learning state
        GraphBuilder* pbuilder;
        std::vector<std::string> md_stack;

        // graph builder operations

        GraphBuilder::GraphBuilder() {}

        GraphBuilder::~GraphBuilder()
        {
            if (is_exported)
            {
                // Figure out how exporting works
            }
            else if (started_export)
            {
                // The exporting process was started, so this'll be weird
            }
            else
            {
                // delete everything
            }
        }

        uint64_t GraphBuilder::edge_lookup(uint64_t edge)
        {
            // following the links while pushing the pointers forward one slot
            // to make future look-ups faster
            EdgeBuilder* curr_edge_builder = edges[edge];
            if (!curr_edge_builder->is_merged)
                return edge;
            edge = curr_edge_builder->tgt_edge;
            EdgeBuilder* prev_edge_builder = curr_edge_builder;
            curr_edge_builder = edges[edge];
            while (curr_edge_builder->is_merged)
            {
                edge = curr_edge_builder->tgt_edge;
                prev_edge_builder->tgt_edge = edge;
                prev_edge_builder = curr_edge_builder;
                curr_edge_builder = edges[edge];
            }
            return edge;
        }

        inline bool GraphBuilder::edge_exists(uint64_t edge)
        {
            return edge * (edge < edges.size());
        }

        inline bool GraphBuilder::node_exists(uint64_t node)
        {
            return node * (node < nodes.size());
        }

        inline bool GraphBuilder::init_exists(uint64_t init)
        {
            return init * (init < inits.size());
        }

        inline bool GraphBuilder::tensor_exists(uint64_t tensor)
        {
            return tensor * (tensor < tensors.size());
        }

        inline bool GraphBuilder::block_exists(uint64_t block)
        {
            return block * (block < blocks.size());
        }

        std::string GraphBuilder::current_mode()
        {
            if (md_stack.size() == 0)
                return "";

            // Not very efficient, but it'll work
            std::stringstream ss;
            ss << md_stack[0];
            for (size_t i = 1; i < md_stack.size(); i++)
                ss << "." << md_stack[i];
            return ss.str();
        }

        bool GraphBuilder::create_edg(Obj& obj, const core::EdgeInfo& info)
        {
            obj.ptr = edges.size();
            edges.push_back(new EdgeBuilder{ .info = info });
            return false;
        }

        bool GraphBuilder::create_tsr(Obj& obj)
        {
            obj.ptr = tensors.size();
            tensors.push_back(new TensorBuilder());
            return false;
        }

        bool GraphBuilder::create_nde(Obj& obj, const std::string& name)
        {
            obj.ptr = nodes.size();
            nodes.push_back(new NodeBuilder{ .name = name });
            return false;
        }

        bool GraphBuilder::create_ini(Obj& obj, const std::string& name)
        {
            obj.ptr = inits.size();
            inits.push_back(new InitBuilder{ .name = name });
            return false;
        }

        bool GraphBuilder::create_blk(Obj& obj, const std::string& name)
        {
            obj.ptr = blocks.size();
            blocks.push_back(new BlockBuilder{ .name = name });
            return false;
        }

        bool GraphBuilder::get_fwd(Obj& obj, uint64_t tensor)
        {
            assert(!is_exported);

            if (!tensor_exists(tensor))
                return error::runtime("Attempted to reference a non-existant tensor");
            obj.ptr = tensors[tensor]->fwd_edge;
            if (!obj.ptr)
                return error::runtime("Attempted to retrieve the uninitialized forward edge of a tensor");
            return false;
        }

        bool GraphBuilder::get_bwd(Obj& obj, uint64_t tensor)
        {
            assert(!is_exported);

            if (!tensor_exists(tensor))
                return error::runtime("Attempted to reference a non-existant tensor");
            obj.ptr = tensors[tensor]->bwd_edge;
            if (!obj.ptr)
                return error::runtime("Attempted to retrieve the uninitialized backward edge of a tensor");
            return false;
        }

        bool GraphBuilder::get_ini(Obj& obj, uint64_t tensor)
        {
            assert(!is_exported);

            if (!tensor_exists(tensor))
                return error::runtime("Attempted to reference a non-existant tensor");
            obj.ptr = tensors[tensor]->init;
            if (!obj.ptr)
                return error::runtime("Attempted to retrieve the uninitialized weight initializer of a tensor");
            return false;
        }

        bool GraphBuilder::set_fwd(uint64_t tensor, uint64_t edge)
        {
            assert(!is_exported);

            if (!tensor_exists(tensor))
                return error::runtime("Attempted to reference a non-existant tensor");
            if (!edge_exists(edge))
                return error::runtime("Attempted to reference a non-existant edge");
            edge = edge_lookup(edge);
            if (tensors[tensor]->bwd_edge)
            {
                // Make sure the shape and fty match
                uint64_t bwd_edge = tensors[tensor]->bwd_edge;
                if (edges[bwd_edge]->info.dims.size() != edges[edge]->info.dims.size())
                    return error::runtime("Tensor rank mismatch found while binding a forward edge");
                for (size_t i = 0; i < edges[edge]->info.dims.size(); i++)
                    if (edges[bwd_edge]->info.dims[i] != edges[edge]->info.dims[i])
                        return error::runtime("Tensor shape mismatch found while binding a forward edge");
                if (edges[bwd_edge]->info.fty != edges[edge]->info.fty)
                    return error::runtime("Tensor fty mismatch found while binding a forward edge");
            }
            tensors[tensor]->fwd_edge = edge;
            return false;
        }

        bool GraphBuilder::set_bwd(uint64_t tensor, uint64_t edge)
        {
            assert(!is_exported);

            if (!tensor_exists(tensor))
                return error::runtime("Attempted to reference a non-existant tensor");
            if (!edge_exists(edge))
                return error::runtime("Attempted to reference a non-existant edge");
            edge = edge_lookup(edge);
            if (tensors[tensor]->fwd_edge)
            {
                // Make sure the shape and fty match
                uint64_t fwd_edge = tensors[tensor]->fwd_edge;
                if (edges[fwd_edge]->info.dims.size() != edges[edge]->info.dims.size())
                    return error::runtime("Tensor rank mismatch found while binding a backward edge");
                for (size_t i = 0; i < edges[edge]->info.dims.size(); i++)
                    if (edges[fwd_edge]->info.dims[i] != edges[edge]->info.dims[i])
                        return error::runtime("Tensor shape mismatch found while binding a backward edge");
                if (edges[fwd_edge]->info.fty != edges[edge]->info.fty)
                    return error::runtime("Tensor fty mismatch found while binding a backward edge");
            }
            tensors[tensor]->bwd_edge = edge;
            return false;
        }

        bool GraphBuilder::set_ini(uint64_t tensor, uint64_t init)
        {
            assert(!is_exported);

            if (!tensor_exists(tensor))
                return error::runtime("Attempted to reference a non-existant tensor");
            if (!init_exists(init))
                return error::runtime("Attempted to reference a non-existant weight initializer");
            tensors[tensor]->init = init;
            return false;

        }

        bool GraphBuilder::mrg(uint64_t lhs_edge, uint64_t rhs_edge)
        {
            assert(!is_exported);
            // Merging rhs into lhs

            if (!edge_exists(lhs_edge) || !edge_exists(rhs_edge))
                return error::runtime("Attempted to reference a non-existant edge object");
            lhs_edge = edge_lookup(lhs_edge);
            rhs_edge = edge_lookup(rhs_edge);
            EdgeBuilder* old_edge = edges[rhs_edge];
            EdgeBuilder* new_edge = edges[lhs_edge];
            assert(!old_edge->is_merged);
            assert(!new_edge->is_merged);

            // Merging the edge inputs
            for (const auto& [md, conn] : old_edge->md_inps)
            {
                if (edges[lhs_edge]->md_inps.contains(md))
                    return error::runtime("Attempted to merge two edges when both had bound inputs");
                assert(nodes[conn.node]->outs.contains(conn.name));
                nodes[conn.node]->outs[conn.name] = lhs_edge;
                edges[lhs_edge]->md_inps[md] = conn;
            }

            // Merging the edge outputs
            for (const auto& [md, conns] : edges[rhs_edge]->md_outs)
                for (const auto& conn : conns)
                {
                    assert(nodes[conn.node]->inps.contains(conn.name));
                    auto& node_inps = nodes[conn.node]->inps.at(conn.name);
                    size_t idx = 0;
                    for (; idx < node_inps.size(); idx++)
                        if (node_inps[idx] == rhs_edge)
                        {
                            node_inps[idx] = lhs_edge;
                            break;
                        }
                    assert(idx != node_inps.size());  // If the node isn't referencing the rhs_edge, that's a bug
                    edges[lhs_edge]->md_outs[md].push_back(conn);  // dupicates will be handled during the export
                }

            old_edge->is_merged = true;
            old_edge->tgt_edge = lhs_edge;

            return false;
        }

        bool GraphBuilder::get_tshp(Obj& obj, ProgramHeap& heap, uint64_t tensor)
        {
            assert(!is_exported);

            if (!tensor_exists(tensor))
                return error::runtime("Attempted toreference a non-existant tensor object");

            // No need to check if the forward and backward edges agree on shape
            // Its guarenteed that they will by the checks done in set_fwd and set_bwd
            size_t edge;
            if (tensors[tensor]->fwd_edge)
                edge = tensors[tensor]->fwd_edge;
            else if (tensors[tensor]->bwd_edge)
                edge = tensors[tensor]->bwd_edge;
            else
                return error::runtime("Unable to retrieve the shape from a fully uninitialized tensor");

            std::vector<Obj> agg_obj;
            for (size_t dim : edges[edge]->info.dims)
            {
                Obj elem_obj;
                if (heap.create_obj_int(elem_obj, dim))
                    return true;
                agg_obj.push_back(elem_obj);
            }
            return heap.create_obj_agg(obj, agg_obj);
        }

        bool GraphBuilder::get_tfty(Obj& obj, ProgramHeap& heap, uint64_t tensor)
        {
            assert(!is_exported);

            if (!tensor_exists(tensor))
                return error::runtime("Attempted toreference a non-existant tensor object");

            // No need to check if the forward and backward edges agree on shape
            // Its guarenteed that they will by the checks done in set_fwd and set_bwd
            if (tensors[tensor]->fwd_edge)
                return heap.create_obj_fty(obj, edges[tensors[tensor]->fwd_edge]->info.fty);
            if (tensors[tensor]->bwd_edge)
                return heap.create_obj_fty(obj, edges[tensors[tensor]->bwd_edge]->info.fty);
            return error::runtime("Unable to retrieve the fty from a fully uninitialized tensor");
        }

        bool GraphBuilder::get_eshp(Obj& obj, ProgramHeap& heap, uint64_t edge)
        {
            assert(!is_exported);

            if (!edge_exists(edge))
                return error::runtime("Attempted toreference a non-existant edge object");
            edge = edge_lookup(edge);

            std::vector<Obj> agg_obj;
            for (size_t dim : edges[edge]->info.dims)
            {
                Obj elem_obj;
                if (heap.create_obj_int(elem_obj, dim))
                    return true;
                agg_obj.push_back(elem_obj);
            }
            return heap.create_obj_agg(obj, agg_obj);
        }

        bool GraphBuilder::get_efty(Obj& obj, ProgramHeap& heap, uint64_t edge)
        {
            assert(!is_exported);

            if (!edge_exists(edge))
                return error::runtime("Attempted to reference a non-existant edge object");
            edge = edge_lookup(edge);
            
            return heap.create_obj_fty(obj, edges[edge]->info.fty);
        }

        bool GraphBuilder::get_einp(Obj& obj, ProgramHeap& heap, uint64_t edge)
        {
            assert(!is_exported);

            if (!edge_exists(edge))
                return error::runtime("Attempted to reference a non-existant edge object");
            edge = edge_lookup(edge);

            return heap.create_obj_bool(obj, node_exists(edges[edge]->md_inps.at(current_mode()).node));
        }

        bool GraphBuilder::add_ndcfg(const std::string& name, uint64_t node, const core::Config& cfg)
        {
            assert(!is_exported);

            if (!node_exists(node))
                return error::runtime("Attempted to reference a non-existant node");
            if (nodes[node]->configs.contains(name))
                return error::runtime("Attempted to overwrite node configuration '%'", name);
            nodes[node]->configs[name] = cfg;
            return false;
        }

        bool GraphBuilder::add_bkcfg(const std::string& name, uint64_t block, const core::Config& cfg)
        {
            assert(!is_exported);

            if (!block_exists(block))
                return error::runtime("Attempted to reference a non-existant block");
            if (blocks[block]->configs.contains(name))
                return error::runtime("Attempted to overwrite block configuration '%'", name);
            blocks[block]->configs[name] = cfg;
            return false;
        }

        bool GraphBuilder::add_incfg(const std::string& name, uint64_t init, const core::Config& cfg)
        {
            assert(!is_exported);

            if (!init_exists(init))
                return error::runtime("Attempted to reference a non-existant weight initializer");
            if (inits[init]->configs.contains(name))
                return error::runtime("Attempted to overwrite init configuration '%'", name);
            inits[init]->configs[name] = cfg;
            return false;
        }

        bool GraphBuilder::set_ndprt(uint64_t node, uint64_t parent)
        {
            assert(!is_exported);

            if (!node_exists(node))
                return error::runtime("Attempted to reference a non-existant node");
            if (!block_exists(parent))
                return error::runtime("Attempted to reference a non-existant block");

            if (nodes[node]->parent)
                return error::runtime("Attempted to set a block's parent when it has already been set");
            nodes[node]->parent = parent;
            
            // Adding the node as a sub-node of parent
            std::string base = nodes[node]->name;
            for (size_t i = 0; true; i++)
            {
                std::string name = base + "~" + std::to_string(i);
                if (!blocks[parent]->sub_nodes.contains(name))
                {
                    blocks[parent]->sub_nodes[name] = node;
                    return false;
                }
            }
            assert(false);
            return error::runtime("Internal error: unreachable code execution in set_ndprt");
        }

        bool GraphBuilder::set_ndinp(const std::string& name, uint64_t node, uint64_t edge)
        {
            assert(!is_exported);

            if (!node_exists(node))
                return error::runtime("Attempted to reference a non-existant node");
            if (!edge_exists(edge))
                return error::runtime("Attempted to reference a non-existant edge");
            edge = edge_lookup(edge);
            // Don't check for nodes[node]->inps.contains(name) here, cause binding to the same
            // node input multiple times is allowed since that's how variadic inputs are created
            NodeBuilder* builder = nodes[node];
            if (std::find(builder->inp_order.begin(), builder->inp_order.end(), name) == builder->inp_order.end())
                builder->inp_order.push_back(name);
            builder->inps[name].push_back(edge);
            edges[edge]->md_outs[current_mode()].push_back({ node, name });
            return false;
        }

        bool GraphBuilder::set_ndout(const std::string& name, uint64_t node, uint64_t edge)
        {
            assert(!is_exported);

            if (!node_exists(node))
                return error::runtime("Attempted to reference a non-existant node");
            if (!edge_exists(edge))
                return error::runtime("Attempted to reference a non-existant edge");
            edge = edge_lookup(edge);
            if (nodes[node]->outs.contains(name))
                return error::runtime("Attempted to bind node output '%' multiple times", name);
            std::string md = current_mode();
            if (edges[edge]->md_inps.contains(md))
                return error::runtime("Attempted to bind an edge input multiple times");
            nodes[node]->out_order.push_back(name);
            nodes[node]->outs[name] = edge;
            edges[edge]->md_inps[md] = { node, name };
            return false;
        }

        bool GraphBuilder::set_bkprt(uint64_t block, uint64_t parent)
        {
            assert(!is_exported);

            if (!block_exists(block) || parent >= blocks.size())  // parent can be null
                return error::runtime("Attempted to reference a non-existant block");

            if (blocks[block]->parent)
                return error::runtime("Attempted to set a block's parent when it has already been set");
            else if (block == root)
                return error::runtime("Attempted to set the root block's parent");

            blocks[block]->parent = parent;
            if (parent)
            {
                // Adding the block as a sub-block of the parent
                std::string base = blocks[block]->name;
                for (size_t i = 0; true; i++)
                {
                    std::string name = base + "~" + std::to_string(i);
                    if (!blocks[parent]->sub_blocks.contains(name))
                    {
                        blocks[parent]->sub_blocks[name] = block;
                        return false;
                    }
                }
                // Should be unreachable
                assert(false);
                return error::runtime("Internal error: unreachable code execution in set_bkprt");
            }

            if (root)
                return error::runtime("Attempted to set a block as the root, when a root was already set");
            root = block;
            return false;
        }

        bool GraphBuilder::set_bkinp(const std::string& name, uint64_t block, uint64_t tensor)
        {
            assert(!is_exported);

            if (!block_exists(block))
                return error::runtime("Attempted to reference a non-existant block");
            if (!tensor_exists(tensor))
                return error::runtime("Attempted to reference a non-existant tensor");

            if (blocks[block]->inps.contains(name))
                return error::runtime("Attempted to bind block input '%' multiple times", name);
            if (!tensors[tensor]->fwd_edge)
                return error::runtime("Attempted to bind block input '%' to a tensor with an uninitialized forward edge", name);
            if (!tensors[tensor]->bwd_edge)
                return error::runtime("Attempted to bind block input '%' to a tensor with an uninitialized backward edge", name);
            blocks[block]->inps[name] = tensor;
            return false;
        }

        bool GraphBuilder::set_bkout(const std::string& name, uint64_t block, uint64_t tensor)
        {
            assert(!is_exported);

            if (!block_exists(block))
                return error::runtime("Attempted to reference a non-existant block");
            if (!tensor_exists(tensor))
                return error::runtime("Attempted to reference a non-existant tensor");

            if (blocks[block]->outs.contains(name))
                return error::runtime("Attempted to bind block output '%' multiple times", name);
            if (tensors[tensor]->fwd_edge == 0)
                return error::runtime("Attempted to bind block output '%' to a tensor with an uninitialized forward edge", name);
            if (tensors[tensor]->bwd_edge == 0)
                return error::runtime("Attempted to bind block output '%' to a tensor with an uninitialized backward edge", name);
            blocks[block]->outs[name] = tensor;
            return false;
        }

        bool GraphBuilder::set_bkext(const std::string& name, uint64_t block, uint64_t tensor)
        {
            assert(!is_exported);

            if (!block_exists(block))
                return error::runtime("Attempted to reference a non-existant block");
            if (!tensor_exists(tensor))
                return error::runtime("Attempted to reference a non-existant tensor");

            if (blocks[block]->exts.contains(name))
                return error::runtime("Attempted to overwrite block weight '%'", name);
            if (tensors[tensor]->fwd_edge == 0)
                return error::runtime("Attempted to extern a tensor with an uninitialized forward edge");
            if (tensors[tensor]->bwd_edge == 0)
                return error::runtime("Attempted to extern a tensor with an uninitialized backward edge");
            blocks[block]->exts[name] = tensor;
            return false;
        }

        bool GraphBuilder::set_bkexp(const std::string& name, uint64_t block, uint64_t tensor)
        {
            assert(!is_exported);

            if (!block_exists(block))
                return error::runtime("Attempted to reference a non-existant block");
            if (!tensor_exists(tensor))
                return error::runtime("Attempted to reference a non-existant tensor");

            if (blocks[block]->exps.contains(name))
                return error::runtime("Attempted to overwrite model export '%'", name);
            if (tensors[tensor]->fwd_edge == 0)
                return error::runtime("Attempted to export a tensor with an uninitialized forward edge");
            if (tensors[tensor]->bwd_edge == 0)
                return error::runtime("Attempted to export a tensor with an uninitialized backward edge");
            blocks[block]->exps[name] = tensor;
            return false;
        }

        bool GraphBuilder::export_graph(core::Graph& graph)
        {
            assert(!is_exported);

            started_export = true;
            if (!block_exists(root))
                return error::graph("The root block of the graph was undefined");

            // First do the exporting of blocks
            if (export_block(graph.model, root))
                return true;

            // Initializing edges[i]->edge and binding the edges to the tensors
            for (uint64_t i = 1; i < edges.size(); i++)
                if (!edges[i]->is_merged && export_edge(edges[i]))
                    return true;
            // Initializing nodes[i]->node, binding the edges to the nodes, and binding the nodes to the blocks
            for (uint64_t i = 1; i < nodes.size(); i++)
                if (export_node(nodes[i]))
                    return true;
            // Binding the nodes to the edges
            for (uint64_t i = 1; i < edges.size(); i++)
                if (!edges[i]->is_merged && bind_edge(edges[i]))
                    return true;

            // Then attach the blocks to the nodes and edges
            if (bind_block(graph.model, root))
                return true;

            // Search through each of the blocks recursively to get all of the
            // inputs, outputs, exports, and model weights
            for (const auto& [name, tensor] : blocks[root]->inps)
                graph.inps[name] = tensors[tensor]->tensor;
            for (const auto& [name, tensor] : blocks[root]->outs)
                graph.outs[name] = tensors[tensor]->tensor;
            if (export_io(graph, "", root))
                return true;

            is_exported = true;
            return false;
        }

        bool GraphBuilder::export_edge(EdgeBuilder* edge)
        {
            edge->edge = new core::Edge();
            
            // Binding the edge to any tensors that referenced it
            for (const auto [id, fwd_edge] : edge->tensors)
            {
                assert(tensor_exists(id));
                assert(tensors[id]->ty != TensorBuilder::NON_EXPORTED);
                // Doesn't matter if its a tensor or parameter, both have a forward and backward edge
                // And the unions used have the same layout for the edges.
                if (fwd_edge)
                    tensors[id]->tensor.forward = edge->edge;
                else
                    tensors[id]->tensor.backward = edge->edge;
            }

            // Copying the edge info (fp and shape)
            edge->edge->info = edge->info;
            return false;
        }

        bool GraphBuilder::export_node(NodeBuilder* node)
        {
            node->node = new core::Node();

            // Binding the node to the parent block
            if (!block_exists(node->parent))
                return error::graph("Found invalid block during graph exporting");
            assert(blocks[node->parent]->block);  // The blocks should all be exported at this point
            blocks[node->parent]->block->sub_nodes[node->name] = node->node;
            node->node->parent = blocks[node->parent]->block;

            // Binding the node inputs
            for (const auto& [name, edge_ids] : node->inps)
            {
                for (uint64_t edge_id : edge_ids)
                {
                    if (!edge_exists(edge_id))
                        return error::graph("Found an invalid edge during graph exporting");
                    node->node->inps[name].push_back(edges[edge_id]->edge);
                }
                node->node->inp_order.push_back(name);
            }

            // Binding the node outputs
            for (const auto& [name, edge_id] : node->outs)
            {
                if (!edge_exists(edge_id))
                    return error::graph("Found an invalid edge during graph exporting");
                node->node->outs[name] = edges[edge_id]->edge;
                node->node->out_order.push_back(name);
            }

            // Configurations
            node->node->name = node->name;
            for (auto& [name, cfg] : node->configs)
                node->node->configs[name] = std::move(cfg);
            return false;
        }

        bool GraphBuilder::bind_edge(EdgeBuilder* edge)
        {
            // Binding the inputs
            for (const auto& [md, conn] : edge->md_inps)
            {
                if (!node_exists(conn.node))
                    return error::graph("Found an invalid node during graph exporting");
                edge->edge->md_inps[md] = { nodes[conn.node]->node, conn.name };
            }

            // Binding the outputs
            for (const auto& [name, conns] : edge->md_outs)
            {
                std::vector<core::Edge::OutConnector> real_conns;
                for (const auto& [node, conn_name] : conns)
                {
                    if (!node_exists(node))
                        return error::graph("Found an invalid node during graph exporting");
                    assert(nodes[node]->node);  // Everything should be exported at this point
                    real_conns.push_back({ nodes[node]->node, conn_name, real_conns.size() });
                }
                edge->edge->md_outs[name] = std::move(real_conns);
            }

            return false;
        }

        bool GraphBuilder::export_init(core::Init*& init, uint64_t i)
        {
            if (!init_exists(i))
                return error::graph("Found an invalid weight initializer during graph exporting");
            init = new core::Init();

            init->name = inits[i]->name;
            for (const auto& [name, config] : inits[i]->configs)
                // I can't guarentee with the current bytecode specification that the inits won't be shared
                // So to make sure that nothing weird happens I need to copy out each one of the configs rather than move them
                init->configs[name] = inits[i]->configs.at(name);  // copy assignment
            return false;
        }

        bool GraphBuilder::export_tensor(uint64_t i)
        {
            // Initial checks
            if (!tensor_exists(i))
                return error::graph("Found an invalid tensor during exporting");
            if (tensors[i]->ty != TensorBuilder::NON_EXPORTED)  // Checking if its already been exported
                return false;
            if (!edge_exists(tensors[i]->fwd_edge) || !edge_exists(tensors[i]->bwd_edge))
                return error::graph("Found a tensor with a missing forward or backward edge");
            tensors[i]->fwd_edge = edge_lookup(tensors[i]->fwd_edge);
            tensors[i]->bwd_edge = edge_lookup(tensors[i]->bwd_edge);

            // Telling the edges about which tensor they belong to
            edges[tensors[i]->fwd_edge]->tensors.push_back({ i, true });
            edges[tensors[i]->bwd_edge]->tensors.push_back({ i, false });

            // Exporting as a tensor or parameter depending on whether the init was specified
            if (tensors[i]->init)
            {
                // Parameter exporting
                tensors[i]->param = {
                    .forward  = nullptr,
                    .backward = nullptr,
                    .data     = nullptr
                };
                tensors[i]->param.init = new core::Init();
                if (export_init(tensors[i]->param.init, tensors[i]->init))
                    return false;
                tensors[i]->ty = TensorBuilder::PARAMETER;
                return false;
            }

            // Tensor exporting
            tensors[i]->tensor = {
                .forward  = nullptr,
                .backward = nullptr
            };
            tensors[i]->ty = TensorBuilder::TENSOR;
            return false;
        }

        bool GraphBuilder::export_block(core::Block& block, uint64_t i)
        {
            if (!block_exists(i))
                return error::graph("Found an invalid block during exporting");
            assert(!blocks[i]->block);  // Asserting that 'i' hasn't already been exported
            blocks[i]->block = &block;

            // making sure all the tensor are exported
            // This will also give type to the edges about which tensors they're bound to
            for (const auto& [name, tensor] : blocks[i]->inps)
            {
                if (export_tensor(tensor))
                    return true;
                assert(tensors[tensor]->ty != TensorBuilder::NON_EXPORTED);
                // Its ok for the input to be a parameter, just treat it like its a tensor anyway
                if (block.inps.contains(name))
                    return error::graph("Found an input name conflict for block %, name %", block.name, name);
                block.inps[name] = tensors[tensor]->tensor;
            }
            for (const auto& [name, tensor] : blocks[i]->outs)
            {
                if (export_tensor(tensor))
                    return true;
                assert(tensors[tensor]->ty != TensorBuilder::NON_EXPORTED);
                if (block.outs.contains(name))
                    return error::graph("Found an output name conflict for block %, name %", block.name, name);
                block.outs[name] = tensors[tensor]->tensor;
            }
            for (const auto& [name, tensor] : blocks[i]->exps)
            {
                if (export_tensor(tensor))
                    return true;
                assert(tensors[tensor]->ty != TensorBuilder::NON_EXPORTED);
                if (block.exports.contains(name))
                    return error::graph("Found an export name conflict for block %, name %", block.name, name);
                block.exports[name] = tensors[tensor]->tensor;
            }
            for (const auto& [name, tensor] : blocks[i]->exts)
            {
                if (export_tensor(tensor))
                    return true;
                if (tensors[tensor]->ty != TensorBuilder::PARAMETER)
                    return error::graph("Found a tensor set as a block weight.  A parameter was expected");
                if (block.weights.contains(name))
                    return error::graph("Found a weight name conflict for block %, name %", block.name, name);
                block.weights[name] = tensors[tensor]->param;
            }

            // The parent should already be initialized for non-root blocks
            if (blocks[i]->parent)
                block.parent = blocks[blocks[i]->parent]->block;
            else
                block.parent = nullptr;  // root block

            // Recursive portion (sub blocks)
            for (const auto& [name, block_id] : blocks[i]->sub_blocks)
            {
                core::Block* sub_block = new core::Block();
                if (export_block(*sub_block, block_id))
                {
                    delete sub_block;
                    for (const auto& [del_name, del_block] : block.sub_blocks)
                        delete del_block;  // This should recursively delete all the exported things
                    return true;
                }
                block.sub_blocks[name] = sub_block;
            }

            // Configurations
            block.name = blocks[i]->name;
            for (auto& [name, cfg] : blocks[i]->configs)
                block.configs[name] = std::move(cfg);

            return false;
        }

        bool GraphBuilder::bind_block(core::Block& block, uint64_t i)
        {
            if (!block_exists(i))
                return error::graph("Found an invalid block during exporting");
            
            // Everything should already be exported at this point
            // the only think left to do is bind the blocks to the tensor data
            for (const auto& [name, tensor] : blocks[i]->inps)
                block.inps[name] = tensors[tensor]->tensor;
            for (const auto& [name, tensor] : blocks[i]->outs)
                block.outs[name] = tensors[tensor]->tensor;
            for (const auto& [name, tensor] : blocks[i]->exps)
                block.exports[name] = tensors[tensor]->tensor;
            for (const auto& [name, tensor] : blocks[i]->exts)
                block.weights[name] = tensors[tensor]->param;
            return false;
        }

        bool GraphBuilder::export_io(core::Graph& graph, const std::string& prefix, uint64_t block)
        {
            for (const auto& [name, tensor] : blocks[block]->exps)
                graph.exports[prefix + name] = tensors[tensor]->tensor;
            for (const auto& [name, tensor] : blocks[block]->exts)
                graph.weights[prefix + name] = tensors[tensor]->param;
            for (const auto& [name, sub_block] : blocks[block]->sub_blocks)
                if (export_io(graph, prefix + name + ".", sub_block))
                    return true;
            return false;
        }

        // stack operations

        bool CallStack::pop(Obj& obj)
        {
            if (sp == 0)
                return error::runtime("Stack pointer out of bounds during pop operation");
            obj = stack[--sp];
            return false;
        }

        bool CallStack::del(size_t i)
        {
            if (i >= sp)
                return error::runtime("Attempted to delete a non-existent stack element");
            sp--;
            for (size_t j = sp - i; j < sp; j++)
                stack[j] = stack[j + 1];
            return false;
        }

        bool CallStack::get(size_t i, Obj& obj)
        {
            if (i >= sp)
                return error::runtime("Attempted to retrieve a non-existent stack element");
            obj = stack[sp - i - 1];
            return false;
        }

        bool CallStack::push(Obj obj)
        {
            if (sp >= stack.size())
                return error::runtime("Stack overflow error");
            stack[sp++] = obj;
            return false;
        }

        // helper functions

        inline bool set_pc(size_t val)
        {
            pc = val;
            return false;
        }

        inline bool push_pc()
        {
            pc_stack.push_back(pc);
            return false;
        }

        inline bool pop_pc()
        {
            assert(pc_stack.size() > 0);
            pc = pc_stack.back();
            pc_stack.pop_back();
            return false;
        }

        inline bool push_md(const std::string& mode)
        {
            md_stack.push_back(mode);
            return false;
        }

        inline bool pop_md()
        {
            if (md_stack.size() == 0)
                return error::runtime("Attempted to release a non-existent evaluation mode");
            md_stack.pop_back();
            return false;
        }

        // instruction implementations

        inline bool exec_jmp()
        {
            return
                set_pc(oprand);
        }

        inline bool exec_brt(CallStack& stack)
        {
            Obj obj;
            return
                stack.pop(obj) ||
                set_pc(*obj.bool_obj ? oprand : pc + sizeof(size_t));
        }

        inline bool exec_brf(CallStack& stack)
        {
            Obj obj;
            return
                stack.pop(obj) ||
                set_pc(*obj.bool_obj ? pc + sizeof(size_t) : oprand);
        }

        inline bool exec_new(CallStack& stack)
        {
            return
                stack.push(data[oprand]) ||
                set_pc(pc + sizeof(size_t));
        }

        inline bool exec_agg(CallStack& stack, ProgramHeap& heap)
        {
            std::vector<Obj> objs;
            objs.reserve(oprand);
            for (size_t i = 0; i < oprand; i++)
            {
                Obj obj;
                if (stack.pop(obj))
                    return true;
                objs.push_back(obj);
            }
            std::reverse(objs.begin(), objs.end());
            Obj dst;
            return
                heap.create_obj_agg(dst, objs) ||
                stack.push(dst) ||
                set_pc(pc + sizeof(size_t));
        }

        inline bool exec_arr(CallStack& stack, ProgramHeap& heap)
        {
            Obj elem_ty, arr_ty;
            return
                stack.pop(elem_ty) ||
                heap.create_type_arr(arr_ty, elem_ty.type_obj) ||
                stack.push(arr_ty);
        }

        inline bool exec_aty(CallStack& stack, ProgramHeap& heap)
        {
            std::vector<TypeObj*> tys;
            tys.reserve(oprand);
            for (size_t i = 0; i < oprand; i++)
            {
                Obj ty;
                if (stack.pop(ty))
                    return true;
                tys.push_back(ty.type_obj);
            }
            std::reverse(tys.begin(), tys.end());
            Obj type;
            return
                heap.create_type_agg(type, tys) ||
                stack.push(type) ||
                set_pc(pc + sizeof(size_t));
        }

        inline bool exec_nul(CallStack& stack)
        {
            return stack.push({ .ptr = 0 });
        }

        inline bool exec_pop(CallStack& stack)
        {
            return
                stack.del(oprand) ||
                set_pc(pc + sizeof(size_t));
        }

        inline bool exec_dup(CallStack& stack)
        {
            Obj obj;
            return
                stack.get(oprand, obj) ||
                stack.push(obj) ||
                set_pc(pc + sizeof(size_t));
        }

        inline bool exec_cpy(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, src, dst;
            return
                stack.pop(type) ||
                stack.pop(src) ||
                type.type_obj->cpy(heap, dst, src) ||
                stack.push(dst);
        }

        inline bool exec_inst(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, inst;
            return
                stack.pop(type) ||
                type.type_obj->inst(heap, inst) ||
                stack.push(inst);
        }

        inline bool exec_call(CallStack& stack)
        {
            Obj proc;
            return
                stack.pop(proc) ||
                push_pc() ||
                set_pc(proc.ptr);
        }

        inline bool exec_ret(CallStack& stack)
        {
            if (pc_stack.size() > 0)
                return pop_pc();

            complete = true;
            return false;
        }

        inline bool exec_set(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->set(heap, lhs, rhs);
        }

        inline bool exec_iadd(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->iadd(heap, lhs, rhs);
        }

        inline bool exec_isub(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->isub(heap, lhs, rhs);
        }

        inline bool exec_imul(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->imul(heap, lhs, rhs);
        }

        inline bool exec_idiv(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->idiv(heap, lhs, rhs);
        }

        inline bool exec_imod(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->imod(heap, lhs, rhs);
        }

        inline bool exec_ipow(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->ipow(heap, lhs, rhs);
        }

        inline bool exec_add(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->add(heap, dst, lhs, rhs) ||
                stack.push(dst);
        }

        inline bool exec_sub(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->sub(heap, dst, lhs, rhs) ||
                stack.push(dst);
        }

        inline bool exec_mul(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->mul(heap, dst, lhs, rhs) ||
                stack.push(dst);
        }

        inline bool exec_div(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->div(heap, dst, lhs, rhs) ||
                stack.push(dst);
        }

        inline bool exec_mod(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->mod(heap, dst, lhs, rhs) ||
                stack.push(dst);
        }

        inline bool exec_pow(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->pow(heap, dst, lhs, rhs) ||
                stack.push(dst);
        }

        inline bool exec_neg(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, src, dst;
            return
                stack.pop(type) ||
                stack.pop(src) ||
                type.type_obj->neg(heap, dst, src) ||
                stack.push(dst);
        }

        inline bool exec_lnot(CallStack& stack, ProgramHeap& heap)
        {
            Obj src, dst;
            return
                stack.pop(src) ||
                heap.create_obj_bool(dst, !src.bool_obj) ||
                stack.push(dst);
        }

        inline bool exec_land(CallStack& stack, ProgramHeap& heap)
        {
            Obj rhs, lhs, dst;
            return
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                heap.create_obj_bool(dst, lhs.bool_obj && rhs.bool_obj) ||
                stack.push(dst);
        }

        inline bool exec_lor(CallStack& stack, ProgramHeap& heap)
        {
            Obj rhs, lhs, dst;
            return
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                heap.create_obj_bool(dst, lhs.bool_obj || rhs.bool_obj) ||
                stack.push(dst);
        }

        inline bool exec_eq(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->eq(heap, dst, lhs, rhs) ||
                stack.push(dst);
        }

        inline bool exec_ne(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->ne(heap, dst, lhs, rhs) ||
                stack.push(dst);
        }

        inline bool exec_gt(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->gt(heap, dst, lhs, rhs) ||
                stack.push(dst);
        }

        inline bool exec_lt(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->lt(heap, dst, lhs, rhs) ||
                stack.push(dst);
        }

        inline bool exec_ge(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->ge(heap, dst, lhs, rhs) ||
                stack.push(dst);
        }

        inline bool exec_le(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->le(heap, dst, lhs, rhs) ||
                stack.push(dst);
        }

        inline bool exec_idx(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->idx(heap, dst, lhs, rhs) ||
                stack.push(dst);
        }

        inline bool exec_len(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, src, dst;
            return
                stack.pop(type) ||
                stack.pop(src) ||
                type.type_obj->len(heap, dst, src) ||
                stack.push(dst);
        }

        inline bool exec_xcfg(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, src, dst;
            return
                stack.pop(type) ||
                stack.pop(src) ||
                type.type_obj->xcfg(heap, dst, src) ||
                stack.push(dst);
        }

        inline bool exec_xstr(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, src, dst;
            return
                stack.pop(type) ||
                stack.pop(src) ||
                type.type_obj->xstr(heap, dst, src) ||
                stack.push(dst);
        }

        inline bool exec_xflt(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, src, dst;
            return
                stack.pop(type) ||
                stack.pop(src) ||
                type.type_obj->xflt(heap, dst, src) ||
                stack.push(dst);
        }

        inline bool exec_xint(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, src, dst;
            return
                stack.pop(type) ||
                stack.pop(src) ||
                type.type_obj->xint(heap, dst, src) ||
                stack.push(dst);
        }

        inline bool exec_dsp(CallStack& stack)
        {
            Obj obj;
            if (stack.pop(obj))
                return true;
            std::cout << *obj.str_obj << std::endl;
            return false;
        }

        inline bool exec_err(CallStack& stack)
        {
            Obj obj;
            if (stack.pop(obj))
                return true;
            return error::runtime(*obj.str_obj);
        }

        inline bool exec_edg(CallStack& stack)
        {
            Obj argnum;
            if (stack.pop(argnum))
                return true;
            std::vector<size_t> dims;
            dims.reserve(*argnum.int_obj);
            for (IntObj i = 0; i < *argnum.int_obj; i++)
            {
                Obj dim;
                if (stack.pop(dim))
                    return true;
                dims.push_back(*dim.int_obj);
            }
            std::reverse(dims.begin(), dims.end());
            Obj obj, fty;
            return
                stack.pop(fty) ||
                pbuilder->create_edg(obj, core::EdgeInfo{ *fty.fty_obj, dims }) ||
                stack.push(obj);
        }

        inline bool exec_tsr(CallStack& stack)
        {
            Obj obj;
            return
                pbuilder->create_tsr(obj) ||
                stack.push(obj);
        }

        inline bool exec_nde(CallStack& stack)
        {
            Obj obj, name;
            return
                stack.pop(name) ||
                pbuilder->create_nde(obj, *name.str_obj) ||
                stack.push(obj);
        }

        inline bool exec_ini(CallStack& stack)
        {
            Obj obj, name;
            return
                stack.pop(name) ||
                pbuilder->create_ini(obj, *name.str_obj) ||
                stack.push(obj);
        }

        inline bool exec_blk(CallStack& stack)
        {
            Obj obj, name;
            return
                stack.pop(name) ||
                pbuilder->create_blk(obj, *name.str_obj) ||
                stack.push(obj);
        }

        inline bool exec_gfwd(CallStack& stack)
        {
            Obj obj, ten;
            return
                stack.pop(ten) ||
                pbuilder->get_fwd(obj, ten.ptr) ||
                stack.push(obj);
        }

        inline bool exec_gbwd(CallStack& stack)
        {
            Obj obj, ten;
            return
                stack.pop(ten) ||
                pbuilder->get_bwd(obj, ten.ptr) ||
                stack.push(obj);
        }

        inline bool exec_gini(CallStack& stack)
        {
            Obj obj, ten;
            return
                stack.pop(ten) ||
                pbuilder->get_ini(obj, ten.ptr) ||
                stack.push(obj);
        }

        inline bool exec_sfwd(CallStack& stack)
        {
            Obj tensor, edge;
            return
                stack.pop(edge) ||
                stack.pop(tensor) ||
                pbuilder->set_fwd(tensor.ptr, edge.ptr) ||
                stack.push(tensor);
        }

        inline bool exec_sbwd(CallStack& stack)
        {
            Obj tensor, edge;
            return
                stack.pop(edge) ||
                stack.pop(tensor) ||
                pbuilder->set_bwd(tensor.ptr, edge.ptr) ||
                stack.push(tensor);
        }

        inline bool exec_sini(CallStack& stack)
        {
            Obj tensor, init;
            return
                stack.pop(init) ||
                stack.pop(tensor) ||
                pbuilder->set_ini(tensor.ptr, init.ptr) ||
                stack.push(tensor);
        }

        inline bool exec_mrg(CallStack& stack)
        {
            Obj lhs, rhs;
            return
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                pbuilder->mrg(lhs.ptr, rhs.ptr);
        }

        inline bool exec_tshp(CallStack& stack, ProgramHeap& heap)
        {
            Obj tensor, shape;
            return
                stack.pop(tensor) ||
                pbuilder->get_tshp(shape, heap, tensor.ptr) ||
                stack.push(shape);
        }

        inline bool exec_tfty(CallStack& stack, ProgramHeap& heap)
        {
            Obj tensor, fty;
            return
                stack.pop(tensor) ||
                pbuilder->get_tfty(fty, heap, tensor.ptr) ||
                stack.push(fty);
        }

        inline bool exec_eshp(CallStack& stack, ProgramHeap& heap)
        {
            Obj edge, shape;
            return
                stack.pop(edge) ||
                pbuilder->get_eshp(shape, heap, edge.ptr) ||
                stack.push(shape);
        }

        inline bool exec_efty(CallStack& stack, ProgramHeap& heap)
        {
            Obj edge, fty;
            return
                stack.pop(edge) ||
                pbuilder->get_efty(fty, heap, edge.ptr) ||
                stack.push(fty);
        }

        inline bool exec_einp(CallStack& stack, ProgramHeap& heap)
        {
            Obj edge, ret;
            return
                stack.pop(edge) ||
                pbuilder->get_einp(ret, heap, edge.ptr) ||
                stack.push(ret);
        }

        inline bool exec_ndcfg(CallStack& stack)
        {
            Obj node, name, obj;
            return
                stack.pop(name) ||
                stack.pop(obj) ||
                stack.pop(node) ||
                pbuilder->add_ndcfg(*name.str_obj, node.ptr, *obj.cfg_obj) ||
                stack.push(node);
        }

        inline bool exec_bkcfg(CallStack& stack)
        {
            Obj block, name, obj;
            return
                stack.pop(name) ||
                stack.pop(obj) ||
                stack.pop(block) ||
                pbuilder->add_bkcfg(*name.str_obj, block.ptr, *obj.cfg_obj) ||
                stack.push(block);
        }

        inline bool exec_incfg(CallStack& stack)
        {
            Obj init, name, obj;
            return
                stack.pop(name) ||
                stack.pop(obj) ||
                stack.pop(init) ||
                pbuilder->add_incfg(*name.str_obj, init.ptr, *obj.cfg_obj) ||
                stack.push(init);
        }

        inline bool exec_ndprt(CallStack& stack)
        {
            Obj node, parent;
            return
                stack.pop(node) ||
                stack.pop(parent) ||
                pbuilder->set_ndprt(node.ptr, parent.ptr) ||
                stack.push(node);
        }

        inline bool exec_ndinp(CallStack& stack)
        {
            Obj name, node, edge;
            return
                stack.pop(name) ||
                stack.pop(edge) ||
                stack.pop(node) ||
                pbuilder->set_ndinp(*name.str_obj, node.ptr, edge.ptr) ||
                stack.push(node);
        }

        inline bool exec_ndout(CallStack& stack)
        {
            Obj name, node, edge;
            return
                stack.pop(name) ||
                stack.pop(edge) ||
                stack.pop(node) ||
                pbuilder->set_ndout(*name.str_obj, node.ptr, edge.ptr) ||
                stack.push(node);
        }

        inline bool exec_bkprt(CallStack& stack)
        {
            Obj block, parent;
            return
                stack.pop(block) ||
                stack.pop(parent) ||
                pbuilder->set_bkprt(block.ptr, parent.ptr) ||
                stack.push(block);
        }

        inline bool exec_bkinp(CallStack& stack)
        {
            Obj name, block, tensor;
            return
                stack.pop(name) ||
                stack.pop(tensor) ||
                stack.pop(block) ||
                pbuilder->set_bkinp(*name.str_obj, block.ptr, tensor.ptr) ||
                stack.push(block);
        }

        inline bool exec_bkout(CallStack& stack)
        {
            Obj name, block, tensor;
            return
                stack.pop(name) ||
                stack.pop(tensor) ||
                stack.pop(block) ||
                pbuilder->set_bkout(*name.str_obj, block.ptr, tensor.ptr) ||
                stack.push(block);
        }

        inline bool exec_bkext(CallStack& stack)
        {
            Obj name, tensor, block;
            return
                stack.pop(name) ||
                stack.pop(tensor) ||
                stack.pop(block) ||
                pbuilder->set_bkext(*name.str_obj, block.ptr, tensor.ptr);
        }

        inline bool exec_bkexp(CallStack& stack)
        {
            Obj name, tensor, block;
            return
                stack.pop(name) ||
                stack.pop(tensor) ||
                stack.pop(block) ||
                pbuilder->set_bkexp(*name.str_obj, block.ptr, tensor.ptr);
        }

        inline bool exec_pshmd(CallStack& stack)
        {
            Obj mode;
            return
                stack.pop(mode) ||
                push_md(*mode.str_obj);
        }

        inline bool exec_popmd(CallStack& stack)
        {
            return
                pop_md();
        }

        bool exec(CallStack& stack, ProgramHeap& heap, GraphBuilder& builder, ByteCode& byte_code, std::string entry_point)
        {
            if (!byte_code.proc_offsets.contains(entry_point))
                return error::runtime("Unable to find entry point '%'", entry_point);

            // Initializing the interpreter state
            pc = byte_code.proc_offsets.at(entry_point);
            code = byte_code.code_segment;
            data = byte_code.data_segment;
            complete = false;
            pc_stack.clear();
            pbuilder = &builder;
            md_stack.clear();
            error::bind_runtime_context(byte_code.debug_info, pc);

            InstructionType ty;
            while (!complete)
            {
                ty = *(InstructionType*)(code + pc);
                pc += sizeof(InstructionType);
                switch (ty)
                {
                case InstructionType::JMP:
                    if (exec_jmp())
                        goto runtime_error;
                    break;
                case InstructionType::BRT:
                    if (exec_brt(stack))
                        goto runtime_error;
                    break;
                case InstructionType::BRF:
                    if (exec_brf(stack))
                        goto runtime_error;
                    break;
                case InstructionType::NEW:
                    if (exec_new(stack))
                        goto runtime_error;
                    break;
                case InstructionType::AGG:
                    if (exec_agg(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::ARR:
                    if (exec_arr(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::ATY:
                    if (exec_aty(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::NUL:
                    if (exec_nul(stack))
                        goto runtime_error;
                    break;
                case InstructionType::POP:
                    if (exec_pop(stack))
                        goto runtime_error;
                    break;
                case InstructionType::DUP:
                    if (exec_dup(stack))
                        goto runtime_error;
                    break;
                case InstructionType::CPY:
                    if (exec_cpy(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::INST:
                    if (exec_inst(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::CALL:
                    if (exec_call(stack))
                        goto runtime_error;
                    break;
                case InstructionType::RET:
                    if (exec_ret(stack))
                        goto runtime_error;
                    break;
                case InstructionType::SET:
                    if (exec_set(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::IADD:
                    if (exec_iadd(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::ISUB:
                    if (exec_isub(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::IMUL:
                    if (exec_imul(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::IDIV:
                    if (exec_idiv(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::IMOD:
                    if (exec_imod(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::IPOW:
                    if (exec_ipow(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::ADD:
                    if (exec_add(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::SUB:
                    if (exec_sub(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::MUL:
                    if (exec_mul(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::DIV:
                    if (exec_div(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::MOD:
                    if (exec_mod(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::POW:
                    if (exec_pow(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::EQ:
                    if (exec_eq(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::NE:
                    if (exec_ne(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::GT:
                    if (exec_gt(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::LT:
                    if (exec_lt(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::GE:
                    if (exec_ge(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::LE:
                    if (exec_le(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::IDX:
                    if (exec_idx(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::LEN:
                    if (exec_len(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::XCFG:
                    if (exec_xcfg(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::XSTR:
                    if (exec_xstr(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::XFLT:
                    if (exec_xflt(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::XINT:
                    if (exec_xint(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::DSP:
                    if (exec_dsp(stack))
                        goto runtime_error;
                    break;
                case InstructionType::ERR:
                    if (exec_err(stack))
                        goto runtime_error;
                    break;

                case InstructionType::EDG:
                    if (exec_edg(stack))
                        goto runtime_error;
                    break;
                case InstructionType::TSR:
                    if (exec_tsr(stack))
                        goto runtime_error;
                    break;
                case InstructionType::NDE:
                    if (exec_nde(stack))
                        goto runtime_error;
                    break;
                case InstructionType::INI:
                    if (exec_ini(stack))
                        goto runtime_error;
                    break;
                case InstructionType::BLK:
                    if (exec_blk(stack))
                        goto runtime_error;
                    break;
                case InstructionType::GFWD:
                    if (exec_gfwd(stack))
                        goto runtime_error;
                    break;
                case InstructionType::GBWD:
                    if (exec_gbwd(stack))
                        goto runtime_error;
                    break;
                case InstructionType::GINI:
                    if (exec_gini(stack))
                        goto runtime_error;
                    break;
                case InstructionType::SFWD:
                    if (exec_sfwd(stack))
                        goto runtime_error;
                    break;
                case InstructionType::SBWD:
                    if (exec_sbwd(stack))
                        goto runtime_error;
                    break;
                case InstructionType::SINI:
                    if (exec_sini(stack))
                        goto runtime_error;
                    break;
                case InstructionType::MRG:
                    if (exec_mrg(stack))
                        goto runtime_error;
                    break;
                case InstructionType::TSHP:
                    if (exec_tshp(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::TFTY:
                    if (exec_tfty(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::ESHP:
                    if (exec_eshp(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::EFTY:
                    if (exec_efty(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::EINP:
                    if (exec_einp(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::NDCFG:
                    if (exec_ndcfg(stack))
                        goto runtime_error;
                    break;
                case InstructionType::BKCFG:
                    if (exec_bkcfg(stack))
                        goto runtime_error;
                    break;
                case InstructionType::INCFG:
                    if (exec_incfg(stack))
                        goto runtime_error;
                    break;
                case InstructionType::NDPRT:
                    if (exec_ndprt(stack))
                        goto runtime_error;
                    break;
                case InstructionType::NDINP:
                    if (exec_ndinp(stack))
                        goto runtime_error;
                    break;
                case InstructionType::NDOUT:
                    if (exec_ndout(stack))
                        goto runtime_error;
                    break;
                case InstructionType::BKPRT:
                    if (exec_bkprt(stack))
                        goto runtime_error;
                    break;
                case InstructionType::BKINP:
                    if (exec_bkinp(stack))
                        goto runtime_error;
                    break;
                case InstructionType::BKOUT:
                    if (exec_bkout(stack))
                        goto runtime_error;
                    break;
                case InstructionType::BKEXT:
                    if (exec_bkext(stack))
                        goto runtime_error;
                    break;
                case InstructionType::BKEXP:
                    if (exec_bkexp(stack))
                        goto runtime_error;
                    break;
                case InstructionType::PSHMD:
                    if (exec_pshmd(stack))
                        goto runtime_error;
                    break;
                case InstructionType::POPMD:
                    if (exec_popmd(stack))
                        goto runtime_error;
                    break;

                default:
                    error::runtime("Invalid instruction opcode '%'", (uint8_t)ty);
                    goto runtime_error;
                }
            }
            return false;

        runtime_error:
            // Unwinding the stack, adding a runtime error at every level
            while (pc_stack.size() != 0)
            {
                exec_ret(stack);
                error::runtime("traceback info");
            }

            return true;
        }
    }
}
