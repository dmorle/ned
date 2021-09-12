#ifndef CUNED_GRAPH_H
#define CUNED_GRAPH_H

#include <ned/core/graph.h>
#include <unordered_set>

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

        class CuGraph
        {
            std::map<std::string, Edge*> inputs;
            std::vector<Edge*> outputs;
            RunId curr_eval;

            std::unordered_set<Edge*> edge_set;
            std::unordered_set<Node*> node_set;

        public:
            CuGraph(const core::Graph* pgraph);
            ~CuGraph();

            void assign_input(std::string& name, void* data, size_t nbytes);
            void eval();
            void get_output(size_t out_num, void* data, size_t nbytes);
        };
    }
}

#endif
