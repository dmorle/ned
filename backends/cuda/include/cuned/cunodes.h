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
            core::tensor_dty inp_dty;
            core::tensor_dty out_dty;
            Edge* inp1;
            Edge* inp2;
            Edge* out;

        public:
            MatMul(std::map<std::string, std::shared_ptr<lang::Obj>>& cargs, core::Edge* inp1, core::Edge* inp2, core::Edge* out);
            virtual void forward(RunId id) override;
            virtual void backward(RunId id) override;
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
