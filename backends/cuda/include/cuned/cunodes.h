#ifndef CUNED_NODES_H
#define CUNED_NODES_H

#include <cuned/cugraph.h>

namespace nn
{
    namespace cuda
    {
        template<typename T>
        class BinOpPointwise :
            public Node
        {
            size_t sz;
            Edge* inp1;
            Edge* inp2;
            Edge* out;

        public:
            BinOpPointwise(size_t sz, Edge* inp1, Edge* inp2, Edge* out) :
                sz(sz), inp1(inp1), inp2(inp2), out(out) {}
            ~BinOpPointwise() { delete inp1; delete inp2; }
        };

        template<typename T>
        class AddPointwise :
            public BinOpPointwise<T>
        {
        public:
            using BinOpPointwise::BinOpPointwise;
            virtual void eval(RunId id) override;
        };

        template<typename T>
        class SubPointwise :
            public BinOpPointwise<T>
        {
        public:
            using BinOpPointwise::BinOpPointwise;
            virtual void eval(RunId id) override;
        };

        template<typename T>
        class MulPointwise :
            public BinOpPointwise<T>
        {
        public:
            using BinOpPointwise::BinOpPointwise;
            virtual void eval(RunId id) override;
        };

        template<typename T>
        class DivPointwise :
            public BinOpPointwise<T>
        {
        public:
            using BinOpPointwise::BinOpPointwise;
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
            BinOpScalar(size_t sz, Edge* inp, Edge* val, Edge* out) :
                sz(sz), inp(inp), val(val), out(out) {}
            ~BinOpScalar() { delete inp; delete val; }
        };

        template<typename T>
        class AddScalar :
            public BinOpScalar<T>
        {
        public:
            using BinOpScalar::BinOpScalar;
            virtual void eval(RunId id) override;
        };

        template<typename T>
        class SubScalar :
            public BinOpScalar<T>
        {
        public:
            using BinOpScalar::BinOpScalar;
            virtual void eval(RunId id) override;
        };

        template<typename T>
        class MulScalar :
            public BinOpScalar<T>
        {
        public:
            using BinOpScalar::BinOpScalar;
            virtual void eval(RunId id) override;
        };

        template<typename T>
        class DivScalar :
            public BinOpScalar<T>
        {
        public:
            using BinOpScalar::BinOpScalar;
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
            public Node
        {
        public:
            using BinOpConst::BinOpConst;
            virtual void eval(RunId id) override;
        };

        template<typename T>
        class SubConst :
            public Node
        {
        public:
            using BinOpConst::BinOpConst;
            virtual void eval(RunId id) override;
        };

        template<typename T>
        class MulConst :
            public Node
        {
        public:
            using BinOpConst::BinOpConst;
            virtual void eval(RunId id) override;
        };

        template<typename T>
        class DivConst :
            public Node
        {
        public:
            using BinOpConst::BinOpConst;
            virtual void eval(RunId id) override;
        };
    }
}

#endif
