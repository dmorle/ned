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
        Edge::Edge()
        {
            data = nullptr;
            id = RunId{};
            dependancy = nullptr;
        }

        Edge::~Edge()
        {
            if (data)
                cudaFree(data);
            if (dependancy)
                delete dependancy;
        }

        void* Edge::get_data(RunId id)
        {
            if (data && this->id == id)
                return data;
            assert(dependancy);
            dependancy->eval(id);
            assert(data);
            assert(this->id == id);
            return data;
        }

        Node* translate_node(const core::Node* pnode)
        {
            Node* pret;
            switch (hash(pnode->name))
            {
            case hash("add_same_intr"):
                if (pnode->name != "add_same_intr") throw GraphError("Unrecognized graph intrinsic name: " + pnode->name);
                assert(pnode->inputs.size() == 2);
                assert(pnode->outputs.size() == 1);
                assert(pnode->outputs[0]->opaque);
                pret = new AddSame(pnode->cargs, pnode->inputs[0], pnode->inputs[1], pnode->outputs[0]);
                break;
            case hash("add_scalar_right_intr"):
                if (pnode->name != "add_scalar_right_intr") throw GraphError("Unrecognized graph intrinsic name: " + pnode->name);
                pnode->cargs;
                assert(pnode->inputs.size() == 2);
                assert(pnode->outputs.size() == 1);
                assert(pnode->outputs[0]->opaque);
                pret = new AddScalar(pnode->cargs, pnode->inputs[0], pnode->inputs[1], pnode->outputs[0]);
                break;
            case hash("add_scalar_left_intr"):
                if (pnode->name != "add_scalar_left_intr") throw GraphError("Unrecognized graph intrinsic name: " + pnode->name);
                pnode->cargs;
                assert(pnode->inputs.size() == 2);
                assert(pnode->outputs.size() == 1);
                assert(pnode->outputs[0]->opaque);
                pret = new AddScalar(pnode->cargs, pnode->inputs[1], pnode->inputs[0], pnode->outputs[0]);
                break;
            case hash("sub_same_intr"):
                if (pnode->name != "sub_same_intr") throw GraphError("Unrecognized graph intrinsic name: " + pnode->name);
                assert(pnode->inputs.size() == 2);
                assert(pnode->outputs.size() == 1);
                assert(pnode->outputs[0]->opaque);
                pret = new SubSame(pnode->cargs, pnode->inputs[0], pnode->inputs[1], pnode->outputs[0]);
                break;
            case hash("sub_scalar_right_intr"):
                if (pnode->name != "sub_scalar_right_intr") throw GraphError("Unrecognized graph intrinsic name: " + pnode->name);
                assert(pnode->inputs.size() == 2);
                assert(pnode->outputs.size() == 1);
                assert(pnode->outputs[0]->opaque);
                pret = new SubScalar(pnode->cargs, pnode->inputs[0], pnode->inputs[1], pnode->outputs[0]);
                break;
            //case hash("sub_scalar_left_intr"):
            //    if (pnode->name != "sub_scalar_left_intr") throw GraphError("Unrecognized graph intrinsic name: " + pnode->name);
            //    assert(pnode->inputs.size() == 2);
            //    assert(pnode->outputs.size() == 1);
            //    assert(pnode->outputs[0]->opaque);
            //    pret = new SubScalar(pnode->cargs,
            //        (Edge**)&pnode->inputs[1]->opaque,
            //        (Edge**)&pnode->inputs[0]->opaque,
            //        (Edge*)pnode->outputs[0]->opaque);
            //    break;
            case hash("mul_same_intr"):
                if (pnode->name != "mul_same_intr") throw GraphError("Unrecognized graph intrinsic name: " + pnode->name);
                assert(pnode->inputs.size() == 2);
                assert(pnode->outputs.size() == 1);
                assert(pnode->outputs[0]->opaque);
                pret = new MulSame(pnode->cargs, pnode->inputs[0], pnode->inputs[1], pnode->outputs[0]);
                break;
            case hash("mul_scalar_right_intr"):
                if (pnode->name != "mul_scalar_right_intr") throw GraphError("Unrecognized graph intrinsic name: " + pnode->name);
                assert(pnode->inputs.size() == 2);
                assert(pnode->outputs.size() == 1);
                assert(pnode->outputs[0]->opaque);
                pret = new MulScalar(pnode->cargs, pnode->inputs[0], pnode->inputs[1], pnode->outputs[0]);
                break;
            case hash("mul_scalar_left_intr"):
                if (pnode->name != "mul_scalar_left_intr") throw GraphError("Unrecognized graph intrinsic name: " + pnode->name);
                assert(pnode->inputs.size() == 2);
                assert(pnode->outputs.size() == 1);
                assert(pnode->outputs[0]->opaque);
                pret = new MulScalar(pnode->cargs, pnode->inputs[1], pnode->inputs[0], pnode->outputs[0]);
                break;
            case hash("div_same_intr"):
                if (pnode->name != "div_same_intr") throw GraphError("Unrecognized graph intrinsic name: " + pnode->name);
                assert(pnode->inputs.size() == 2);
                assert(pnode->outputs.size() == 1);
                assert(pnode->outputs[0]->opaque);
                pret = new DivSame(pnode->cargs, pnode->inputs[1], pnode->inputs[0], pnode->outputs[0]);
                break;
            case hash("div_scalar_right_intr"):
                if (pnode->name != "div_scalar_right_intr") throw GraphError("Unrecognized graph intrinsic name: " + pnode->name);
                assert(pnode->inputs.size() == 2);
                assert(pnode->outputs.size() == 1);
                assert(pnode->outputs[0]->opaque);
                pret = new DivScalar(pnode->cargs, pnode->inputs[1], pnode->inputs[0], pnode->outputs[0]);
                break;
            //case hash("div_scalar_left_intr"):
            //    if (pnode->name != "div_scalar_left_intr") throw GraphError("Unrecognized graph intrinsic name: " + pnode->name);
            //    assert(pnode->inputs.size() == 2);
            //    assert(pnode->outputs.size() == 1);
            //    assert(pnode->outputs[0]->opaque);
            //    pret = new DivScalar(pnode->cargs,
            //        (Edge**)&pnode->inputs[1]->opaque,
            //        (Edge**)&pnode->inputs[0]->opaque,
            //        (Edge*)pnode->outputs[0]->opaque);
            //    break;
            //case hash("sigmoid_intr"):
            //    if (pnode->name != "sigmoid_intr") throw GraphError("Unrecognized graph intrinsic name: " + pnode->name);
            //    break;
            //case hash("tanh_intr"):
            //    if (pnode->name != "tanh_intr") throw GraphError("Unrecognized graph intrinsic name: " + pnode->name);
            //    break;
            //case hash("relu_intr"):
            //    if (pnode->name != "relu_intr") throw GraphError("Unrecognized graph intrinsic name: " + pnode->name);
            //    break;
            //case hash("leaky_relu_intr"):
            //    if (pnode->name != "leaky_relu_intr") throw GraphError("Unrecognized graph intrinsic name: " + pnode->name);
            //    break;
            //case hash("matmul_intr"):
            //    if (pnode->name != "matmul_intr") throw GraphError("Unrecognized graph intrinsic name: " + pnode->name);
            //    assert(pnode->inputs.size() == 2);
            //    assert(pnode->outputs.size() == 1);
            //    assert(pnode->outputs[0]->opaque);
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
                ((Edge*)out->opaque)->dependancy = pret;
            return pret;
        }

        CuGraph::CuGraph(const core::Graph* pgraph)
        {
            // Starting from the graph outputs and going to the graph inputs

        }
    }
}
