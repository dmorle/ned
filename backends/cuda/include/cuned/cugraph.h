#ifndef CUNED_GRAPH_H
#define CUNED_GRAPH_H

#include <ned/core/graph.h>

namespace nn
{
    namespace cuda
    {
        using RunId = uint32_t;

        class Node;
        class Edge
        {
        public:
            void* data;
            RunId id;
            Node* dependancy;

            Edge();
            ~Edge();

            void* get_data(RunId id);
        };

        class Node
        {
        public:
            virtual ~Node() {}

            virtual void eval(RunId id) = 0;
        };

        class CuGraph
        {

        };
    }
}

#endif
