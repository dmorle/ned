#ifndef CUNED_NODES_H
#define CUNED_NODES_H

#include <cuned/cugraph.h>
#include <ned/lang/obj.h>

#include <cuda_runtime.h>

namespace nn
{
    namespace cuda
    {
        template<typename T>
        class BinOpSame :
            public Node
        {
            size_t sz;
            Edge* inp1;
            Edge* inp2;
            Edge* out;

        public:
            BinOpSame(const std::vector<std::shared_ptr<lang::Obj>>& cargs, Edge** pinp1, Edge** pinp2, Edge* out)
            {
                sz = 1;
                for (auto& e : cargs)
                {
                    assert(e->ty == lang::ObjType::INT);
                    sz *= std::static_pointer_cast<lang::ObjInt>(e)->data.val;
                }
                if (!*pinp1)
                {
                    *pinp1 = new Edge();
                    // TODO: check to make sure it was allocated
                    cudaMalloc(&(*pinp1)->data, sz * sizeof(T));
                }
                if (!*pinp2)
                {
                    *pinp2 = new Edge();
                    // TODO: check to make sure it was allocated
                    cudaMalloc(&(*pinp2)->data, sz * sizeof(T));
                }
                inp1 = *pinp1;
                inp2 = *pinp2;
                this->out = out;
            }
            ~BinOpSame()
            {
                // This will cause a double free, I need to find a way to mark the edges as deleted
                delete inp1;
                delete inp2;
            }
        };

        template<typename T>
        class AddSame :
            public BinOpSame<T>
        {
        public:
            using BinOpSame<T>::BinOpSame;
            virtual void eval(RunId id) override;
        };

        template<typename T>
        class SubSame :
            public BinOpSame<T>
        {
        public:
            using BinOpSame<T>::BinOpSame;
            virtual void eval(RunId id) override;
        };

        template<typename T>
        class MulSame :
            public BinOpSame<T>
        {
        public:
            using BinOpSame<T>::BinOpSame;
            virtual void eval(RunId id) override;
        };

        template<typename T>
        class DivSame :
            public BinOpSame<T>
        {
        public:
            using BinOpSame<T>::BinOpSame;
            virtual void eval(RunId id) override;
        };

        template<typename T>
        class BinOpScalar :
            public Node
        {
            size_t sz;
            Edge* inp;
            Edge* val;
            Edge* out;

        public:
            BinOpScalar(const std::vector<std::shared_ptr<lang::Obj>>& cargs, Edge** pinp, Edge** pval, Edge* out)
            {
                sz = 1;
                for (auto& e : cargs)
                {
                    assert(e->ty == lang::ObjType::INT);
                    sz *= std::static_pointer_cast<lang::ObjInt>(e)->data.val;
                }
                if (!*pinp)
                {
                    *pinp = new Edge();
                    // TODO: check to make sure it was allocated
                    cudaMalloc(&(*pinp)->data, sz * sizeof(T));
                }
                if (!*pval)
                {
                    *pval = new Edge();
                    // TODO: check to make sure it was allocated
                    cudaMalloc(&(*pval)->data, sizeof(T));
                }
                inp = *pinp;
                val = *pval;
                this->out = out;
            }
            ~BinOpScalar() { delete inp; delete val; }
        };

        template<typename T>
        class AddScalar :
            public BinOpScalar<T>
        {
        public:
            using BinOpScalar<T>::BinOpScalar;
            virtual void eval(RunId id) override;
        };

        template<typename T>
        class SubScalar :
            public BinOpScalar<T>
        {
        public:
            using BinOpScalar<T>::BinOpScalar;
            virtual void eval(RunId id) override;
        };

        template<typename T>
        class MulScalar :
            public BinOpScalar<T>
        {
        public:
            using BinOpScalar<T>::BinOpScalar;
            virtual void eval(RunId id) override;
        };

        template<typename T>
        class DivScalar :
            public BinOpScalar<T>
        {
        public:
            using BinOpScalar<T>::BinOpScalar;
            virtual void eval(RunId id) override;
        };

        template<typename T>
        class BinOpConst :
            public Node
        {
            size_t sz;
            Edge* inp;
            T val;
            Edge* out;

        public:
            BinOpConst(size_t sz, Edge* inp, T val, Edge* out) :
                sz(sz), inp(inp), val(val), out(out) {}
            ~BinOpConst() { delete inp; }
        };

        template<typename T>
        class AddConst :
            public BinOpConst
        {
        public:
            using BinOpConst<T>::BinOpConst;
            virtual void eval(RunId id) override;
        };

        template<typename T>
        class SubConst :
            public BinOpConst
        {
        public:
            using BinOpConst<T>::BinOpConst;
            virtual void eval(RunId id) override;
        };

        template<typename T>
        class MulConst :
            public BinOpConst
        {
        public:
            using BinOpConst<T>::BinOpConst;
            virtual void eval(RunId id) override;
        };

        template<typename T>
        class DivConst :
            public BinOpConst
        {
        public:
            using BinOpConst<T>::BinOpConst;
            virtual void eval(RunId id) override;
        };

        template<typename T>
        class MatMul :
            public Node
        {
            size_t m;
            size_t s;
            size_t n;
            Edge* inp1;
            Edge* inp2;
            Edge* out;

        public:
            MatMul(const std::vector<std::shared_ptr<lang::Obj>>& cargs, Edge** pinp1, Edge** pinp2, Edge* out)
            {
                assert(cargs.size() == 3);
                assert(cargs[0]->ty == lang::ObjType::INT);
                assert(cargs[1]->ty == lang::ObjType::INT);
                assert(cargs[2]->ty == lang::ObjType::INT);
                m = std::static_pointer_cast<lang::ObjInt>(cargs[0])->data.val;
                s = std::static_pointer_cast<lang::ObjInt>(cargs[1])->data.val;
                n = std::static_pointer_cast<lang::ObjInt>(cargs[2])->data.val;
                if (!*pinp1)
                {
                    *pinp1 = new Edge{};
                    cudaMalloc(&(*pinp1)->data, m * s * sizeof(T));
                }
                if (*pinp2)
                {
                    *pinp2 = new Edge{};
                    cudaMalloc(&(*pinp2)->data, s * n * sizeof(T));
                }
                inp1 = *pinp1;
                inp2 = *pinp2;
                this->out = out;
            }

            virtual void eval(RunId id) override;
        };
    }
}

#endif
