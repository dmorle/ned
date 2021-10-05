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
        inline size_t fwidth_size(core::tensor_dty dty)
        {
            switch (dty)
            {
            case core::tensor_dty::F16:
                return 2;
            case core::tensor_dty::F32:
                return 4;
            case core::tensor_dty::F64:
                return 8;
            }
            return 0;
        }

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
            BinOpSame(std::map<std::string, std::shared_ptr<lang::Obj>>& cargs, core::Edge* inp1, core::Edge* inp2, core::Edge* out);
        };

        class AddSame :
            public BinOpSame
        {
        public:
            using BinOpSame::BinOpSame;
            virtual void forward(RunId id) override;
            virtual void backward(RunId id) override;
        };

        class SubSame :
            public BinOpSame
        {
        public:
            using BinOpSame::BinOpSame;
            virtual void forward(RunId id) override;
            virtual void backward(RunId id) override;
        };

        class MulSame :
            public BinOpSame
        {
        public:
            using BinOpSame::BinOpSame;
            virtual void forward(RunId id) override;
            virtual void backward(RunId id) override;
        };

        class DivSame :
            public BinOpSame
        {
        public:
            using BinOpSame::BinOpSame;
            virtual void forward(RunId id) override;
            virtual void backward(RunId id) override;
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
            BinOpScalar(std::map<std::string, std::shared_ptr<lang::Obj>>& cargs, core::Edge* inp, core::Edge* val, core::Edge* out);
        };

        class AddScalar :
            public BinOpScalar
        {
        public:
            using BinOpScalar::BinOpScalar;
            virtual void forward(RunId id) override;
            virtual void backward(RunId id) override;
        };

        class SubScalar :
            public BinOpScalar
        {
        public:
            using BinOpScalar::BinOpScalar;
            virtual void forward(RunId id) override;
            virtual void backward(RunId id) override;
        };

        class MulScalar :
            public BinOpScalar
        {
        public:
            using BinOpScalar::BinOpScalar;
            virtual void forward(RunId id) override;
            virtual void backward(RunId id) override;
        };

        class DivScalar :
            public BinOpScalar
        {
        public:
            using BinOpScalar::BinOpScalar;
            virtual void forward(RunId id) override;
            virtual void backward(RunId id) override;
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
                    cudaMalloc(&((Edge*)inp1->opaque)->forward_data, m * s * fwidth_size(inp1->dsc.dty));
                    cudaMalloc(&((Edge*)inp1->opaque)->backward_data, m * s * fwidth_size(inp1->dsc.dty));
                }
                if (!inp2->opaque)
                {
                    inp2->opaque = new Edge{};
                    cudaMalloc(&((Edge*)inp2->opaque)->forward_data, s * n * fwidth_size(inp2->dsc.dty));
                    cudaMalloc(&((Edge*)inp2->opaque)->backward_data, s * n * fwidth_size(inp2->dsc.dty));
                }
                this->inp1 = (Edge*)inp1->opaque;
                this->inp2 = (Edge*)inp2->opaque;
                this->out = (Edge*)out->opaque;
                inp1_dty = inp1->dsc.dty;
                inp2_dty = inp2->dsc.dty;
                out_dty = out->dsc.dty;
            }

            virtual void forward(RunId id) override;
        };
    
        class ActivationFn :
            public Node
        {
        protected:
            size_t sz;
            core::tensor_dty dty;
            Edge* inp;
            Edge* out;

        public:
            ActivationFn(std::map<std::string, std::shared_ptr<lang::Obj>>& cargs, core::Edge* inp, core::Edge* out);
        };

        class Sigmoid :
            public ActivationFn
        {
        public:
            using ActivationFn::ActivationFn;
            virtual void forward(RunId id) override;
            virtual void backward(RunId id) override;
        };

        class Tanh :
            public ActivationFn
        {
        public:
            using ActivationFn::ActivationFn;
            virtual void forward(RunId id) override;
            virtual void backward(RunId id) override;
        };

        class ReLU :
            public ActivationFn
        {
        public:
            using ActivationFn::ActivationFn;
            virtual void forward(RunId id) override;
            virtual void backward(RunId id) override;
        };
    }
}

#endif
