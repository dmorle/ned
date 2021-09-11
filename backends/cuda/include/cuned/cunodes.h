#ifndef CUNED_NODES_H
#define CUNED_NODES_H

#include <cuned/cugraph.h>
#include <ned/lang/obj.h>

#include <cassert>

#include <cuda_runtime.h>

namespace nn
{
    namespace cuda
    {
        class BinOpSame :
            public Node
        {
        protected:
            size_t sz;
            core::tensor_dty inp1_dty;
            core::tensor_dty inp2_dty;
            core::tensor_dty out_dty;
            Edge* inp1;
            Edge* inp2;
            Edge* out;

        public:
            BinOpSame(const std::vector<std::shared_ptr<lang::Obj>>& cargs, core::Edge* inp1, core::Edge* inp2, core::Edge* out)
            {
                assert(out->opaque);

                sz = 1;
                for (auto& e : inp1->dsc.dims)
                    sz *= e;
                if (!inp1->opaque)
                {
                    inp1->opaque = new Edge();
                    // TODO: check to make sure it was allocated
                    cudaMalloc(&((Edge*)inp1->opaque)->data, sz * core::dtype_size(inp1->dsc.dty));
                }
                if (!inp2->opaque)
                {
                    inp2->opaque = new Edge();
                    // TODO: check to make sure it was allocated
                    cudaMalloc(&((Edge*)inp2->opaque)->data, sz * core::dtype_size(inp2->dsc.dty));
                }
                this->inp1 = (Edge*)inp1->opaque;
                this->inp2 = (Edge*)inp2->opaque;
                this->out = (Edge*)out->opaque;
                inp1_dty = inp1->dsc.dty;
                inp2_dty = inp2->dsc.dty;
                out_dty = out->dsc.dty;
            }
            ~BinOpSame()
            {
                // This will cause a double free, I need to find a way to mark the edges as deleted
                delete inp1;
                delete inp2;
            }
        };

        class AddSame :
            public BinOpSame
        {
        public:
            using BinOpSame::BinOpSame;
            virtual void eval(RunId id) override;
        };

        class SubSame :
            public BinOpSame
        {
        public:
            using BinOpSame::BinOpSame;
            virtual void eval(RunId id) override;
        };

        class MulSame :
            public BinOpSame
        {
        public:
            using BinOpSame::BinOpSame;
            virtual void eval(RunId id) override;
        };

        class DivSame :
            public BinOpSame
        {
        public:
            using BinOpSame::BinOpSame;
            virtual void eval(RunId id) override;
        };

        class BinOpScalar :
            public Node
        {
        protected:
            size_t sz;
            core::tensor_dty inp_dty;
            core::tensor_dty val_dty;
            core::tensor_dty out_dty;
            Edge* inp;
            Edge* val;
            Edge* out;

        public:
            BinOpScalar(const std::vector<std::shared_ptr<lang::Obj>>& cargs, core::Edge* inp, core::Edge* val, core::Edge* out)
            {
                assert(out->opaque);

                sz = 1;
                for (auto& e : inp->dsc.dims)
                    sz *= e;
                if (!inp->opaque)
                {
                    inp->opaque = new Edge();
                    // TODO: check to make sure it was allocated
                    cudaMalloc(&((Edge*)inp->opaque)->data, sz * core::dtype_size(inp->dsc.dty));
                }
                if (!val->opaque)
                {
                    val->opaque = new Edge();
                    // TODO: check to make sure it was allocated
                    cudaMalloc(&((Edge*)val->opaque)->data, core::dtype_size(val->dsc.dty));
                }
                this->inp = (Edge*)inp->opaque;
                this->val = (Edge*)val->opaque;
                this->out = (Edge*)out->opaque;
                inp_dty = inp->dsc.dty;
                val_dty = val->dsc.dty;
                out_dty = out->dsc.dty;
            }
            ~BinOpScalar() { delete inp; delete val; }
        };

        class AddScalar :
            public BinOpScalar
        {
        public:
            using BinOpScalar::BinOpScalar;
            virtual void eval(RunId id) override;
        };

        class SubScalar :
            public BinOpScalar
        {
        public:
            using BinOpScalar::BinOpScalar;
            virtual void eval(RunId id) override;
        };

        class MulScalar :
            public BinOpScalar
        {
        public:
            using BinOpScalar::BinOpScalar;
            virtual void eval(RunId id) override;
        };

        class DivScalar :
            public BinOpScalar
        {
        public:
            using BinOpScalar::BinOpScalar;
            virtual void eval(RunId id) override;
        };

        class MatMul :
            public Node
        {
            size_t m;
            size_t s;
            size_t n;
            core::tensor_dty inp1_dty;
            core::tensor_dty inp2_dty;
            core::tensor_dty out_dty;
            Edge* inp1;
            Edge* inp2;
            Edge* out;

        public:
            MatMul(const std::vector<std::shared_ptr<lang::Obj>>& cargs, core::Edge* inp1, core::Edge* inp2, core::Edge* out)
            {
                assert(inp1->dsc.dims.size() == 2);
                assert(inp2->dsc.dims.size() == 2);
                m = inp1->dsc.dims[0];
                s = inp1->dsc.dims[1];
                n = out->dsc.dims[1];
                if (!inp1->opaque)
                {
                    inp1->opaque = new Edge{};
                    cudaMalloc(&((Edge*)inp1->opaque)->data, m * s * sizeof(T));
                }
                if (!inp2->opaque)
                {
                    inp2->opaque = new Edge{};
                    cudaMalloc(&((Edge*)inp2->opaque)->data, s * n * sizeof(T));
                }
                this->inp1 = (Edge*)inp1->opaque;
                this->inp2 = (Edge*)inp2->opaque;
                this->out = (Edge*)out->opaque;
                inp1_dty = inp1->dsc.dty;
                inp2_dty = inp2->dsc.dty;
                out_dty = out->dsc.dty;
            }

            virtual void eval(RunId id) override;
        };
    }
}

#endif
