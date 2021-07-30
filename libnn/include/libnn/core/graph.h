#ifndef NN_GRAPH_H
#define NN_GRAPH_H

#include <tuple>
#include <vector>
#include <string>

#include <libnn/core/tensor.h>

namespace nn
{
    class Node;

    class Edge
    {
    public:
        bool is_static;
        tensor_dsc dsc;
        Node* input;
        std::vector<Node*> outputs;
    };

    class Node
    {
	public:
        std::string name;
        std::vector<Edge*> inputs;
        std::vector<Edge*> outputs;
    };

	class Graph
	{
    public:

	};
}

#endif
