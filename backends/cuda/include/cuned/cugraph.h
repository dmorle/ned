#ifndef CUNED_GRAPH_H
#define CUNED_GRAPH_H

#include <ned/core/graph.h>

namespace nn
{
    namespace cuda
    {
        class GraphError :
            public std::exception
        {
        public:
            std::string errmsg;

            GraphError(const std::string& errmsg);
        };

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

        Node* translate_node(const core::Node* pnode);

        class CuGraph
        {
            std::map<std::string, Edge*> inputs;
            std::vector<Edge*> outputs;

        public:
            CuGraph(const core::Graph* pgraph);
            ~CuGraph();
        };
    }
}

#endif
