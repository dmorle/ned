#include <cuned/cugraph.h>
#include <cuned/cunodes.h>
#include <functional>
#include <cassert>

#include <cuda_runtime.h>

#define FNV_PRIME 0x00000100000001B3ULL
#define FNV_OFFSET_BASIS 0XCBF29CE484222325ULL

constexpr size_t hash(const char* s)
{
    size_t h = FNV_OFFSET_BASIS;
    for (const char* c = s; *c; c++)
        h = (h * FNV_PRIME) ^ *c;
    return h;
}

constexpr size_t hash(const std::string& s)
{
    return hash(s.c_str());
}

namespace nn
{
    namespace cuda
    {
        GraphError::GraphError(const std::string& errmsg)
        {
            this->errmsg = errmsg;
        }

        const char* GraphError::what() const
        {
            return errmsg.c_str();
        }

        Edge::Edge()
        {
            forward_data = nullptr;
            backward_data = nullptr;
            forward_id = RunId{};
            dependancy = nullptr;
        }

        Edge::~Edge()
        {
            if (forward_data)
                cudaFree(forward_data);
            if (backward_data)
                cudaFree(backward_data);
        }

        void Edge::forward(RunId id)
        {
            if (forward_data && this->forward_id == id)
                return;
            assert(dependancy);
            dependancy->forward(id);
            assert(forward_data);
            assert(this->forward_id == id);
        }

        void Edge::backward(RunId id)
        {
            if (backward_data && this->backward_id == id)
                return;
            assert(dependancy);
            dependancy->forward(id);
            assert(backward_data);
            assert(this->backward_id == id);
        }

        void translate_node(core::Node* pnode)
        {
            for (auto out : pnode->outputs)
                assert(out->opaque);

            Node* npnode;
            switch (hash(pnode->name))
            {
            case hash("add_same_intr"):
                if (pnode->name != "add_same_intr") throw GraphError("Unrecognized graph intrinsic name: " + pnode->name);
                assert(pnode->inputs.size() == 2);
                assert(pnode->outputs.size() == 1);
                npnode = new AddSame(pnode->cargs, pnode->inputs[0], pnode->inputs[1], pnode->outputs[0]);
                break;
            case hash("add_scalar_right_intr"):
                if (pnode->name != "add_scalar_right_intr") throw GraphError("Unrecognized graph intrinsic name: " + pnode->name);
                pnode->cargs;
                assert(pnode->inputs.size() == 2);
                assert(pnode->outputs.size() == 1);
                npnode = new AddScalar(pnode->cargs, pnode->inputs[0], pnode->inputs[1], pnode->outputs[0]);
                break;
            case hash("add_scalar_left_intr"):
                if (pnode->name != "add_scalar_left_intr") throw GraphError("Unrecognized graph intrinsic name: " + pnode->name);
                pnode->cargs;
                assert(pnode->inputs.size() == 2);
                assert(pnode->outputs.size() == 1);
                npnode = new AddScalar(pnode->cargs, pnode->inputs[1], pnode->inputs[0], pnode->outputs[0]);
                break;
            case hash("sub_same_intr"):
                if (pnode->name != "sub_same_intr") throw GraphError("Unrecognized graph intrinsic name: " + pnode->name);
                assert(pnode->inputs.size() == 2);
                assert(pnode->outputs.size() == 1);
                npnode = new SubSame(pnode->cargs, pnode->inputs[0], pnode->inputs[1], pnode->outputs[0]);
                break;
            case hash("sub_scalar_right_intr"):
                if (pnode->name != "sub_scalar_right_intr") throw GraphError("Unrecognized graph intrinsic name: " + pnode->name);
                assert(pnode->inputs.size() == 2);
                assert(pnode->outputs.size() == 1);
                npnode = new SubScalar(pnode->cargs, pnode->inputs[0], pnode->inputs[1], pnode->outputs[0]);
                break;
            //case hash("sub_scalar_left_intr"):
            //    if (pnode->name != "sub_scalar_left_intr") throw GraphError("Unrecognized graph intrinsic name: " + pnode->name);
            //    assert(pnode->inputs.size() == 2);
            //    assert(pnode->outputs.size() == 1);
            //    pret = new SubScalar(pnode->cargs,
            //        (Edge**)&pnode->inputs[1]->opaque,
            //        (Edge**)&pnode->inputs[0]->opaque,
            //        (Edge*)pnode->outputs[0]->opaque);
            //    break;
            case hash("mul_same_intr"):
                if (pnode->name != "mul_same_intr") throw GraphError("Unrecognized graph intrinsic name: " + pnode->name);
                assert(pnode->inputs.size() == 2);
                assert(pnode->outputs.size() == 1);
                npnode = new MulSame(pnode->cargs, pnode->inputs[0], pnode->inputs[1], pnode->outputs[0]);
                break;
            case hash("mul_scalar_right_intr"):
                if (pnode->name != "mul_scalar_right_intr") throw GraphError("Unrecognized graph intrinsic name: " + pnode->name);
                assert(pnode->inputs.size() == 2);
                assert(pnode->outputs.size() == 1);
                npnode = new MulScalar(pnode->cargs, pnode->inputs[0], pnode->inputs[1], pnode->outputs[0]);
                break;
            case hash("mul_scalar_left_intr"):
                if (pnode->name != "mul_scalar_left_intr") throw GraphError("Unrecognized graph intrinsic name: " + pnode->name);
                assert(pnode->inputs.size() == 2);
                assert(pnode->outputs.size() == 1);
                npnode = new MulScalar(pnode->cargs, pnode->inputs[1], pnode->inputs[0], pnode->outputs[0]);
                break;
            case hash("div_same_intr"):
                if (pnode->name != "div_same_intr") throw GraphError("Unrecognized graph intrinsic name: " + pnode->name);
                assert(pnode->inputs.size() == 2);
                assert(pnode->outputs.size() == 1);
                npnode = new DivSame(pnode->cargs, pnode->inputs[1], pnode->inputs[0], pnode->outputs[0]);
                break;
            case hash("div_scalar_right_intr"):
                if (pnode->name != "div_scalar_right_intr") throw GraphError("Unrecognized graph intrinsic name: " + pnode->name);
                assert(pnode->inputs.size() == 2);
                assert(pnode->outputs.size() == 1);
                npnode = new DivScalar(pnode->cargs, pnode->inputs[1], pnode->inputs[0], pnode->outputs[0]);
                break;
            //case hash("div_scalar_left_intr"):
            //    if (pnode->name != "div_scalar_left_intr") throw GraphError("Unrecognized graph intrinsic name: " + pnode->name);
            //    assert(pnode->inputs.size() == 2);
            //    assert(pnode->outputs.size() == 1);
            //    pret = new DivScalar(pnode->cargs,
            //        (Edge**)&pnode->inputs[1]->opaque,
            //        (Edge**)&pnode->inputs[0]->opaque,
            //        (Edge*)pnode->outputs[0]->opaque);
            //    break;
            case hash("sigmoid_intr"):
                if (pnode->name != "sigmoid_intr") throw GraphError("Unrecognized graph intrinsic name: " + pnode->name);
                assert(pnode->inputs.size() == 1);
                assert(pnode->outputs.size() == 1);
                npnode = new Sigmoid(pnode->cargs, pnode->inputs[0], pnode->outputs[0]);
                break;
            case hash("tanh_intr"):
                if (pnode->name != "tanh_intr") throw GraphError("Unrecognized graph intrinsic name: " + pnode->name);
                assert(pnode->inputs.size() == 1);
                assert(pnode->outputs.size() == 1);
                npnode = new Tanh(pnode->cargs, pnode->inputs[0], pnode->outputs[0]);
                break;
            case hash("relu_intr"):
                if (pnode->name != "relu_intr") throw GraphError("Unrecognized graph intrinsic name: " + pnode->name);
                assert(pnode->inputs.size() == 1);
                assert(pnode->outputs.size() == 1);
                npnode = new ReLU(pnode->cargs, pnode->inputs[0], pnode->outputs[0]);
                break;
            //case hash("leaky_relu_intr"):
            //    if (pnode->name != "leaky_relu_intr") throw GraphError("Unrecognized graph intrinsic name: " + pnode->name);
            //    break;
            //case hash("matmul_intr"):
            //    if (pnode->name != "matmul_intr") throw GraphError("Unrecognized graph intrinsic name: " + pnode->name);
            //    assert(pnode->inputs.size() == 2);
            //    assert(pnode->outputs.size() == 1);
            //    pret = new MatMul(pnode->cargs,
            //        (Edge**)&pnode->inputs[1]->opaque,
            //        (Edge**)&pnode->inputs[0]->opaque,
            //        (Edge*)pnode->outputs[0]->opaque);
            //    break;
            default:
                throw GraphError("Unrecognized graph intrinsic name: " + pnode->name);
            }

            // assigning the edge dependancies
            for (auto out : pnode->outputs)
                ((Edge*)out->opaque)->dependancy = npnode;

            // marking the node as translated
            pnode->opaque = npnode;

            // recursing over the graph
            for (auto inp : pnode->inputs)
                if (inp->input && !inp->input->opaque)  // Not an input edge, and the input node hasn't been translated yet
                    translate_node(inp->input);
        }

        void detach_edge(const core::Edge* pEdge, std::unordered_set<Edge*>& edge_set, std::unordered_set<Node*>& node_set);
        void detach_node(const core::Node* pNode, std::unordered_set<Edge*>& edge_set, std::unordered_set<Node*>& node_set);

        void detach_edge(const core::Edge* pEdge, std::unordered_set<Edge*>& edge_set, std::unordered_set<Node*>& node_set)
        {
            edge_set.insert({ (Edge*)pEdge->opaque });
            pEdge->opaque = nullptr;
            if (pEdge->input && pEdge->input->opaque)  // An input exists and it hasn't already been freed
                detach_node(pEdge->input, edge_set, node_set);
        }

        void detach_node(const core::Node* pNode, std::unordered_set<Edge*>& edge_set, std::unordered_set<Node*>& node_set)
        {
            node_set.insert({ (Node*)pNode->opaque });
            pNode->opaque = nullptr;
            for (auto inp : pNode->inputs)
                if (inp->opaque)
                    detach_edge(inp, edge_set, node_set);
        }

        CuGraph::CuGraph(const core::Graph* pgraph)
        {
            // Creating a new graph off the opaque points of the given
            for (auto out : pgraph->outputs)
            {
                // Creating the output edge
                size_t sz = 1;
                for (auto e : out->dsc.dims)
                    sz *= e;
                Edge* nout = new Edge();
                // TODO: check for allocation failure
                cudaMalloc(&nout->forward_data, sz * core::dtype_size(out->dsc.dty));
                cudaMalloc(&nout->backward_data, sz * core::dtype_size(out->dsc.dty));
                out->opaque = nout;

                // translating all edge dependancies
                if (!out->input->opaque)
                    translate_node(out->input);
            }

            // Stealing pointers to the graph outputs
            for (auto out : pgraph->outputs)
                outputs.push_back((Edge*)out->opaque);
            // Stealing pointers to the graph inputs
            for (auto& [name, inp] : pgraph->inputs)
                inputs[name] = (Edge*)inp->opaque;

            // Detaching the newly created graph from the given graph (setting all the opaque pointers to null)
            // While detaching, all nodes and edges are added to a set for deletion without double freeing in non-trivial topologies
            for (auto out : pgraph->outputs)
                detach_edge(out, this->edge_set, this->node_set);

            curr_eval = RunId{};
        }

        CuGraph::~CuGraph()
        {
            for (auto e : edge_set)
                delete e;
            for (auto e : node_set)
                delete e;
        }

        RunId CuGraph::generate_id()
        {
            return ++curr_eval;
        }

        void CuGraph::assign_input(const std::string& name, void* data, size_t nbytes, RunId forward_id)
        {
            // TODO: check for errors
            cudaMemcpy(inputs[name]->forward_data, data, nbytes, cudaMemcpyKind::cudaMemcpyHostToDevice);
            inputs[name]->forward_id = forward_id;
        }

        void CuGraph::forward(RunId id)
        {
            // Evaluating each edge for the current run
            for (auto out : outputs)
                out->forward(id);
        }

        void CuGraph::backward(RunId id)
        {
            for (auto& [name, inp] : inputs)
                inp->backward(id);
        }

        void CuGraph::get_output(size_t out_num, void* data, size_t nbytes)
        {
            // TODO: check for errors
            cudaMemcpy(data, outputs[out_num]->forward_data, nbytes, cudaMemcpyKind::cudaMemcpyDeviceToHost);
        }
    }
}
