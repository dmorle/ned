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
            case hash("add_same"):
                if (pnode->name != "add_same") throw GraphError("Unrecognized graph node name: " + pnode->name);
                assert(pnode->inputs.size() == 2);
                assert(pnode->outputs.size() == 1);
                assert(pnode->outputs[0]->opaque);
                pret = new AddSame<float>(pnode->cargs,
                    (Edge**)&pnode->inputs[0]->opaque,
                    (Edge**)&pnode->inputs[1]->opaque,
                    (Edge*)pnode->outputs[0]->opaque);
                break;
            case hash("add_scalar_right"):
                if (pnode->name != "add_scalar_right") throw GraphError("Unrecognized graph node name: " + pnode->name);
                pnode->cargs;
                assert(pnode->inputs.size() == 2);
                assert(pnode->outputs.size() == 1);
                assert(pnode->outputs[0]->opaque);
                pret = new AddScalar<float>(pnode->cargs,
                    (Edge**)&pnode->inputs[0]->opaque,
                    (Edge**)&pnode->inputs[1]->opaque,
                    (Edge*)pnode->outputs[0]->opaque);
            case hash("add_scalar_left"):
                if (pnode->name != "add_scalar_left") throw GraphError("Unrecognized graph node name: " + pnode->name);
                pnode->cargs;
                assert(pnode->inputs.size() == 2);
                assert(pnode->outputs.size() == 1);
                assert(pnode->outputs[0]->opaque);
                pret = new AddScalar<float>(pnode->cargs,
                    (Edge**)&pnode->inputs[1]->opaque,
                    (Edge**)&pnode->inputs[0]->opaque,
                    (Edge*)pnode->outputs[0]->opaque);
            case hash("sub_same"):
                if (pnode->name != "sub_same") throw GraphError("Unrecognized graph node name: " + pnode->name);
                assert(pnode->inputs.size() == 2);
                assert(pnode->outputs.size() == 1);
                assert(pnode->outputs[0]->opaque);
                pret = new SubSame<float>(pnode->cargs,
                    (Edge**)&pnode->inputs[0]->opaque,
                    (Edge**)&pnode->inputs[1]->opaque,
                    (Edge*)pnode->outputs[0]->opaque);
                break;
            case hash("sub_scalar_right"):
                if (pnode->name != "sub_scalar_right") throw GraphError("Unrecognized graph node name: " + pnode->name);
                assert(pnode->inputs.size() == 2);
                assert(pnode->outputs.size() == 1);
                assert(pnode->outputs[0]->opaque);
                pret = new SubScalar<float>(pnode->cargs,
                    (Edge**)&pnode->inputs[0]->opaque,
                    (Edge**)&pnode->inputs[1]->opaque,
                    (Edge*)pnode->outputs[0]->opaque);
                break;
            case hash("sub_scalar_left"):
                if (pnode->name != "sub_scalar_left") throw GraphError("Unrecognized graph node name: " + pnode->name);
                assert(pnode->inputs.size() == 2);
                assert(pnode->outputs.size() == 1);
                assert(pnode->outputs[0]->opaque);
                pret = new SubScalar<float>(pnode->cargs,
                    (Edge**)&pnode->inputs[1]->opaque,
                    (Edge**)&pnode->inputs[0]->opaque,
                    (Edge*)pnode->outputs[0]->opaque);
                break;
            case hash("mul_same"):
                if (pnode->name != "mul_same") throw GraphError("Unrecognized graph node name: " + pnode->name);
                assert(pnode->inputs.size() == 2);
                assert(pnode->outputs.size() == 1);
                assert(pnode->outputs[0]->opaque);
                pret = new MulSame<float>(pnode->cargs,
                    (Edge**)&pnode->inputs[0]->opaque,
                    (Edge**)&pnode->inputs[1]->opaque,
                    (Edge*)pnode->outputs[0]->opaque);
                break;
            case hash("mul_scalar_right"):
                if (pnode->name != "mul_scalar_right") throw GraphError("Unrecognized graph node name: " + pnode->name);
                assert(pnode->inputs.size() == 2);
                assert(pnode->outputs.size() == 1);
                assert(pnode->outputs[0]->opaque);
                pret = new MulScalar<float>(pnode->cargs,
                    (Edge**)&pnode->inputs[0]->opaque,
                    (Edge**)&pnode->inputs[1]->opaque,
                    (Edge*)pnode->outputs[0]->opaque);
                break;
            case hash("mul_scalar_left"):
                if (pnode->name != "mul_scalar_left") throw GraphError("Unrecognized graph node name: " + pnode->name);
                assert(pnode->inputs.size() == 2);
                assert(pnode->outputs.size() == 1);
                assert(pnode->outputs[0]->opaque);
                pret = new MulScalar<float>(pnode->cargs,
                    (Edge**)&pnode->inputs[1]->opaque,
                    (Edge**)&pnode->inputs[0]->opaque,
                    (Edge*)pnode->outputs[0]->opaque);
                break;
            case hash("div_same"):
                if (pnode->name != "div_same") throw GraphError("Unrecognized graph node name: " + pnode->name);
                assert(pnode->inputs.size() == 2);
                assert(pnode->outputs.size() == 1);
                assert(pnode->outputs[0]->opaque);
                pret = new DivSame<float>(pnode->cargs,
                    (Edge**)&pnode->inputs[0]->opaque,
                    (Edge**)&pnode->inputs[1]->opaque,
                    (Edge*)pnode->outputs[0]->opaque);
                break;
            case hash("div_scalar_right"):
                if (pnode->name != "div_scalar_right") throw GraphError("Unrecognized graph node name: " + pnode->name);
                assert(pnode->inputs.size() == 2);
                assert(pnode->outputs.size() == 1);
                assert(pnode->outputs[0]->opaque);
                pret = new DivScalar<float>(pnode->cargs,
                    (Edge**)&pnode->inputs[0]->opaque,
                    (Edge**)&pnode->inputs[1]->opaque,
                    (Edge*)pnode->outputs[0]->opaque);
                break;
            case hash("div_scalar_left"):
                if (pnode->name != "div_scalar_left") throw GraphError("Unrecognized graph node name: " + pnode->name);
                assert(pnode->inputs.size() == 2);
                assert(pnode->outputs.size() == 1);
                assert(pnode->outputs[0]->opaque);
                pret = new DivScalar<float>(pnode->cargs,
                    (Edge**)&pnode->inputs[1]->opaque,
                    (Edge**)&pnode->inputs[0]->opaque,
                    (Edge*)pnode->outputs[0]->opaque);
                break;
            case hash("sigmoid"):
                if (pnode->name != "sigmoid") throw GraphError("Unrecognized graph node name: " + pnode->name);
                break;
            case hash("tanh"):
                if (pnode->name != "tanh") throw GraphError("Unrecognized graph node name: " + pnode->name);
                break;
            case hash("relu"):
                if (pnode->name != "relu") throw GraphError("Unrecognized graph node name: " + pnode->name);
                break;
            case hash("leaky_relu"):
                if (pnode->name != "leaky_relu") throw GraphError("Unrecognized graph node name: " + pnode->name);
                break;
            case hash("matmul"):
                if (pnode->name != "matmul") throw GraphError("Unrecognized graph node name: " + pnode->name);
                assert(pnode->inputs.size() == 2);
                assert(pnode->outputs.size() == 1);
                assert(pnode->outputs[0]->opaque);
                pret = new MatMul<float>(pnode->cargs,
                    (Edge**)&pnode->inputs[1]->opaque,
                    (Edge**)&pnode->inputs[0]->opaque,
                    (Edge*)pnode->outputs[0]->opaque);
                break;
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
