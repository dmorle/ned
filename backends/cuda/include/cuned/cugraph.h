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

            const char* what() const override;
        };

        using RunId = uint32_t;

        class Node;
        class Edge
        {
        public:
            void* forward_data;
            void* backward_data;
            RunId forward_id;
            RunId backward_id;
            Node* dependancy;

            Edge();
            ~Edge();

            void forward(RunId id);
            void backward(RunId id);
        };

        class Node
        {
        public:
            virtual ~Node() {}

            virtual void forward(RunId id) = 0;
            virtual void backward(RunId id) = 0;
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

            RunId generate_id();
            
            void assign_input(const std::string& name, void* data, size_t nbytes, RunId id);
            void get_output(size_t out_num, void* data, size_t nbytes);

            void forward(RunId id);
            void backward(RunId id);
        };
    }
}

#endif
